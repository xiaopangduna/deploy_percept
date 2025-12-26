#include "deploy_percept/post_process/YoloV5DetectPostProcess.hpp"
#include "deploy_percept/post_process/types.hpp"
#include <vector>
#include <algorithm>
#include <set>
#include <cstring>

namespace deploy_percept {
namespace post_process {

YoloV5DetectPostProcess::YoloV5DetectPostProcess(const YoloV5DetectPostProcess::Params& params)
    : YoloBasePostProcess(), params_(params) {
    // 初始化锚框值，从参数中获取
    if (params_.anchors.size() >= 3) {
        anchor0_ = params_.anchors[0];
        anchor1_ = params_.anchors[1];
        anchor2_ = params_.anchors[2];
    } else {
        // 默认锚框值
        anchor0_ = {10, 13, 16, 30, 33, 23};
        anchor1_ = {30, 61, 62, 45, 59, 119};
        anchor2_ = {116, 90, 156, 198, 373, 326};
    }
}

int YoloV5DetectPostProcess::process(
    int8_t* input0,
    int8_t* input1, 
    int8_t* input2,
    int model_in_h,
    int model_in_w,
    BoxRect pads,
    float scale_w,
    float scale_h,
    std::vector<int32_t>& qnt_zps,
    std::vector<float>& qnt_scales,
    DetectResultGroup* group
) {
    memset(group, 0, sizeof(DetectResultGroup));

    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;

    // stride 8
    int stride0 = 8;
    int grid_h0 = model_in_h / stride0;
    int grid_w0 = model_in_w / stride0;
    int validCount0 = 0;
    const int prop_box_size = (5 + params_.obj_class_num);  // 重新计算box_size
    validCount0 = processYoloOutput(input0, anchor0_.data(), grid_h0, grid_w0, 
                                   model_in_h, model_in_w, stride0, filterBoxes, objProbs,
                                   classId, params_.conf_threshold, qnt_zps[0], qnt_scales[0]);

    // stride 16
    int stride1 = 16;
    int grid_h1 = model_in_h / stride1;
    int grid_w1 = model_in_w / stride1;
    int validCount1 = 0;
    validCount1 = processYoloOutput(input1, anchor1_.data(), grid_h1, grid_w1, 
                                   model_in_h, model_in_w, stride1, filterBoxes, objProbs,
                                   classId, params_.conf_threshold, qnt_zps[1], qnt_scales[1]);

    // stride 32
    int stride2 = 32;
    int grid_h2 = model_in_h / stride2;
    int grid_w2 = model_in_w / stride2;
    int validCount2 = 0;
    validCount2 = processYoloOutput(input2, anchor2_.data(), grid_h2, grid_w2, 
                                   model_in_h, model_in_w, stride2, filterBoxes, objProbs,
                                   classId, params_.conf_threshold, qnt_zps[2], qnt_scales[2]);

    int validCount = validCount0 + validCount1 + validCount2;
    // no object detect
    if (validCount <= 0) {
        return 0;
    }

    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i) {
        indexArray.push_back(i);
    }

    // 使用与main.cpp中相同的快速排序方法
    quickSortIndices(objProbs, 0, validCount - 1, indexArray);

    std::set<int> class_set(classId.begin(), classId.end());

    for (auto c : class_set) {
        nms(validCount, filterBoxes, classId, indexArray, c, params_.nms_threshold);
    }

    int last_count = 0;
    group->count = 0;
    /* box valid detect target */
    for (int i = 0; i < validCount; ++i) {
        if (indexArray[i] == -1 || last_count >= params_.obj_numb_max_size) {
            continue;
        }
        int n = indexArray[i];

        float x1 = filterBoxes[n * 4 + 0] - pads.left;
        float y1 = filterBoxes[n * 4 + 1] - pads.top;
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];

        int id = classId[n];
        float obj_conf = objProbs[i];

        // 使用与main.cpp一致的坐标计算方式
        group->results[last_count].box.left = static_cast<int>(clamp(x1, 0, model_in_w) / scale_w);
        group->results[last_count].box.top = static_cast<int>(clamp(y1, 0, model_in_h) / scale_h);
        group->results[last_count].box.right = static_cast<int>(clamp(x2, 0, model_in_w) / scale_w);
        group->results[last_count].box.bottom = static_cast<int>(clamp(y2, 0, model_in_h) / scale_h);
        group->results[last_count].prop = obj_conf;
        
        // 设置标签名称，这里只是框架，实际需要从外部加载标签
        snprintf(group->results[last_count].name, params_.obj_name_max_size, "class_%d", id);

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
    const int prop_box_size = (5 + params_.obj_class_num);  // 使用参数中的类别数计算box_size
    
    for (int a = 0; a < 3; a++) {
        for (int i = 0; i < grid_h; i++) {
            for (int j = 0; j < grid_w; j++) {
                int8_t box_confidence = input[(prop_box_size * a + 4) * grid_len + i * grid_w + j];
                if (box_confidence >= thres_i8) {
                    int offset = (prop_box_size * a) * grid_len + i * grid_w + j;
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
                    for (int k = 1; k < params_.obj_class_num; ++k) {  // 使用参数中的类别数量
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

} // namespace post_process
} // namespace deploy_percept