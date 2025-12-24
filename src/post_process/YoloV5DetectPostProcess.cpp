#include "deploy_percept/post_process/YoloV5DetectPostProcess.hpp"
#include <vector>
#include <algorithm>
#include <set>
#include <cstring>

namespace deploy_percept {
namespace post_process {

YoloV5DetectPostProcess::YoloV5DetectPostProcess(float conf_threshold, float nms_threshold)
    : YoloBasePostProcess(conf_threshold, nms_threshold) {
    // 初始化锚框值
    static const int default_anchor0[6] = {10, 13, 16, 30, 33, 23};
    static const int default_anchor1[6] = {30, 61, 62, 45, 59, 119};
    static const int default_anchor2[6] = {116, 90, 156, 198, 373, 326};
    
    anchor0 = default_anchor0;
    anchor1 = default_anchor1;
    anchor2 = default_anchor2;
}

int YoloV5DetectPostProcess::process(
    int8_t* input0,
    int8_t* input1, 
    int8_t* input2,
    int model_in_h,
    int model_in_w,
    BOX_RECT pads,
    float scale_w,
    float scale_h,
    std::vector<int32_t>& qnt_zps,
    std::vector<float>& qnt_scales,
    detect_result_group_t* group
) {
    memset(group, 0, sizeof(detect_result_group_t));

    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;

    // stride 8
    int stride0 = 8;
    int grid_h0 = model_in_h / stride0;
    int grid_w0 = model_in_w / stride0;
    int validCount0 = 0;
    validCount0 = processYoloOutput(input0, const_cast<int*>(anchor0), grid_h0, grid_w0, 
                                   model_in_h, model_in_w, stride0, filterBoxes, objProbs,
                                   classId, conf_threshold_, qnt_zps[0], qnt_scales[0]);

    // stride 16
    int stride1 = 16;
    int grid_h1 = model_in_h / stride1;
    int grid_w1 = model_in_w / stride1;
    int validCount1 = 0;
    validCount1 = processYoloOutput(input1, const_cast<int*>(anchor1), grid_h1, grid_w1, 
                                   model_in_h, model_in_w, stride1, filterBoxes, objProbs,
                                   classId, conf_threshold_, qnt_zps[1], qnt_scales[1]);

    // stride 32
    int stride2 = 32;
    int grid_h2 = model_in_h / stride2;
    int grid_w2 = model_in_w / stride2;
    int validCount2 = 0;
    validCount2 = processYoloOutput(input2, const_cast<int*>(anchor2), grid_h2, grid_w2, 
                                   model_in_h, model_in_w, stride2, filterBoxes, objProbs,
                                   classId, conf_threshold_, qnt_zps[2], qnt_scales[2]);

    int validCount = validCount0 + validCount1 + validCount2;
    // no object detect
    if (validCount <= 0) {
        return 0;
    }

    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i) {
        indexArray.push_back(i);
    }

    // 为 YoloV5DetectPostProcess 提供的排序方法
    // 由于我们不能直接访问 YoloV5DetectPostProcess 的方法，这里使用标准库的排序
    std::vector<std::pair<float, int>> scoreIndexPairs;
    for (int i = 0; i < objProbs.size(); ++i) {
        scoreIndexPairs.push_back({objProbs[i], i});
    }
    std::sort(scoreIndexPairs.begin(), scoreIndexPairs.end(), 
              [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                  return a.first > b.first;
              });
    
    for (int i = 0; i < scoreIndexPairs.size(); ++i) {
        indexArray[i] = scoreIndexPairs[i].second;
    }

    std::set<int> class_set(classId.begin(), classId.end());

    for (auto c : class_set) {
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold_);
    }

    int last_count = 0;
    group->count = 0;
    /* box valid detect target */
    for (int i = 0; i < validCount; ++i) {
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) {
            continue;
        }
        int n = indexArray[i];

        float x1 = filterBoxes[n * 4 + 0] - pads.left;
        float y1 = filterBoxes[n * 4 + 1] - pads.top;
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];
        float obj_conf = objProbs[i];

        group->results[last_count].box.left = YoloBasePostProcess::clamp(x1 / scale_w, 0, model_in_w);
        group->results[last_count].box.top = YoloBasePostProcess::clamp(y1 / scale_h, 0, model_in_h);
        group->results[last_count].box.right = YoloBasePostProcess::clamp(x2 / scale_w, 0, model_in_w);
        group->results[last_count].box.bottom = YoloBasePostProcess::clamp(y2 / scale_h, 0, model_in_h);
        group->results[last_count].prop = obj_conf;
        
        // 设置标签名称，这里只是框架，实际需要从外部加载标签
        snprintf(group->results[last_count].name, OBJ_NAME_MAX_SIZE, "class_%d", id);

        last_count++;
    }
    group->count = last_count;

    return 0;
}

int YoloV5DetectPostProcess::processYoloOutput(int8_t* input, int* anchor, int grid_h, int grid_w, 
                                              int height, int width, int stride,
                                              std::vector<float>& boxes, std::vector<float>& objProbs, 
                                              std::vector<int>& classId, float threshold,
                                              int32_t zp, float scale) {
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t thres_i8 = qntF32ToAffine(threshold, zp, scale);
    
    for (int a = 0; a < 3; a++) {
        for (int i = 0; i < grid_h; i++) {
            for (int j = 0; j < grid_w; j++) {
                int8_t box_confidence = input[(YoloBasePostProcess::PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
                if (box_confidence >= thres_i8) {
                    int offset = (YoloBasePostProcess::PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
                    int8_t *in_ptr = input + offset;
                    float box_x = (deqntAffineToF32(*in_ptr, zp, scale)) * 2.0 - 0.5;
                    float box_y = (deqntAffineToF32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
                    float box_w = (deqntAffineToF32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
                    float box_h = (deqntAffineToF32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = box_w * box_w * (float)anchor[a * 2];
                    box_h = box_h * box_h * (float)anchor[a * 2 + 1];
                    box_x -= (box_w / 2.0);
                    box_y -= (box_h / 2.0);

                    int8_t maxClassProbs = in_ptr[5 * grid_len];
                    int maxClassId = 0;
                    for (int k = 1; k < OBJ_CLASS_NUM; ++k) {
                        int8_t prob = in_ptr[(5 + k) * grid_len];
                        if (prob > maxClassProbs) {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }
                    if (maxClassProbs > thres_i8) {
                        objProbs.push_back((deqntAffineToF32(maxClassProbs, zp, scale)) * (deqntAffineToF32(box_confidence, zp, scale)));
                        classId.push_back(maxClassId);
                        validCount++;
                        boxes.push_back(box_x);
                        boxes.push_back(box_y);
                        boxes.push_back(box_w);
                        boxes.push_back(box_h);
                    }
                }
            }
        }
    }
    return validCount;
}

void YoloV5DetectPostProcess::quickSortIndices(std::vector<float>& input, int left, int right, std::vector<int>& indices) {
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right) {
        key_index = indices[left];
        key = input[left];
        while (low < high) {
            while (low < high && input[high] <= key) {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key) {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quickSortIndices(input, left, low - 1, indices);
        quickSortIndices(input, low + 1, right, indices);
    }
}

int8_t YoloV5DetectPostProcess::qntF32ToAffine(float f32, int32_t zp, float scale) {
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)clip(dst_val, -128, 127);
    return res;
}

float YoloV5DetectPostProcess::deqntAffineToF32(int8_t qnt, int32_t zp, float scale) {
    return ((float)qnt - (float)zp) * scale;
}

int32_t YoloV5DetectPostProcess::clip(float val, float min, float max) {
    float f = val <= min ? min : (val >= max ? max : val);
    return static_cast<int32_t>(f);
}

} // namespace post_process
} // namespace deploy_percept