#include "deploy_percept/post_process/YoloBasePostProcess.hpp"
#include "deploy_percept/post_process/types.hpp"
#include <algorithm>
#include <set>
#include <vector>
#include <cmath>
#include <malloc.h>
#include <opencv2/opencv.hpp>

namespace deploy_percept {
namespace post_process {

YoloBasePostProcess::YoloBasePostProcess() {
    // 无状态构造函数，不需要任何初始化
}

// clamp函数实现
int YoloBasePostProcess::clamp(float val, int min, int max) {
    return val > min ? (val < max ? static_cast<int>(val) : max) : min;
}

// 先定义被其他函数调用的基础函数
float YoloBasePostProcess::CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, 
                                           float xmin1, float ymin1, float xmax1, float ymax1) {
    float w = std::fmax(0.f, std::fmin(xmax0, xmax1) - std::fmax(xmin0, xmin1) + 1.0);
    float h = std::fmax(0.f, std::fmin(ymax0, ymax1) - std::fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

int32_t YoloBasePostProcess::clip(float val, float min, float max) {
    float f = val <= min ? min : (val >= max ? max : val);
    return static_cast<int32_t>(f);
}

// 然后定义依赖上述函数的函数
int8_t YoloBasePostProcess::qntF32ToAffine(float f32, int32_t zp, float scale) {
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)YoloBasePostProcess::clip(dst_val, -128, 127);
    return res;
}

int YoloBasePostProcess::retainHighestScoringBoxesByNMS(int validCount, std::vector<float>& outputLocations, 
                         std::vector<int> classIds, std::vector<int>& order, 
                         int filterId, float threshold) {
    for (int i = 0; i < validCount; ++i) {
        int n = order[i];
        if (n == -1 || classIds[n] != filterId) {
            continue;
        }
        for (int j = i + 1; j < validCount; ++j) {
            int m = order[j];
            if (m == -1 || classIds[m] != filterId) {
                continue;
            }
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = YoloBasePostProcess::CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold) {
                order[j] = -1;
            }
        }
    }
    return 0;
}

float YoloBasePostProcess::deqntAffineToF32(int8_t qnt, int32_t zp, float scale) {
    return ((float)qnt - (float)zp) * scale;
}

int YoloBasePostProcess::quickSortIndices(std::vector<float> &input, int left, int right, std::vector<int> &indices) {
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
    return low;
}

void YoloBasePostProcess::computeSegMask(std::vector<float> &A, float *B, uint8_t *C, int ROWS_A, int COLS_A, int COLS_B)
{
    float temp = 0;
    for (int i = 0; i < ROWS_A; i++)
    {
        for (int j = 0; j < COLS_B; j++)
        {
            temp = 0;
            for (int k = 0; k < COLS_A; k++)
            {
                temp += A[i * COLS_A + k] * B[k * COLS_B + j];
            }
            if (temp > 0)
            {
                C[i * COLS_B + j] = 4;
            }
            else
            {
                C[i * COLS_B + j] = 0;
            }
        }
    }
}

void YoloBasePostProcess::resizeSegMasks(uint8_t *input_image, int input_width, int input_height, int boxes_num,
                                          uint8_t *output_image, int target_width, int target_height)
{
    for (int b = 0; b < boxes_num; b++)
    {
        cv::Mat src_image(input_height, input_width, CV_8U, &input_image[b * input_width * input_height]);
        cv::Mat dst_image;
        cv::resize(src_image, dst_image, cv::Size(target_width, target_height), 0, 0, cv::INTER_LINEAR);
        memcpy(&output_image[b * target_width * target_height], dst_image.data, target_width * target_height * sizeof(uint8_t));
    }
}

void YoloBasePostProcess::mergeBoxMasks(uint8_t *seg_mask, uint8_t *all_mask_in_one, float *boxes, int boxes_num,
                                         int *cls_id, int height, int width)
{
    for (int b = 0; b < boxes_num; b++)
    {
        float x1 = boxes[b * 4 + 0];
        float y1 = boxes[b * 4 + 1];
        float x2 = boxes[b * 4 + 2];
        float y2 = boxes[b * 4 + 3];

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                if (j >= x1 && j < x2 && i >= y1 && i < y2)
                {
                    if (all_mask_in_one[i * width + j] == 0)
                    {
                        if (seg_mask[b * width * height + i * width + j] > 0)
                        {
                            all_mask_in_one[i * width + j] = (cls_id[b] + 1);
                        }
                        else
                        {
                            all_mask_in_one[i * width + j] = 0;
                        }
                    }
                }
            }
        }
    }
}

void YoloBasePostProcess::seg_reverse(uint8_t *seg_mask, uint8_t *cropped_seg, uint8_t *seg_mask_real,
                                       int input_image_height, int input_image_width, int cropped_height, int cropped_width,
                                       int ori_in_height, int ori_in_width, int y_pad, int x_pad)
{
    if (y_pad == 0 && x_pad == 0 && ori_in_height == input_image_height && ori_in_width == input_image_width)
    {
        memcpy(seg_mask_real, seg_mask, ori_in_height * ori_in_width);
        return;
    }

    int cropped_index = 0;
    for (int i = 0; i < input_image_height; i++)
    {
        for (int j = 0; j < input_image_width; j++)
        {
            if (i >= y_pad && i < input_image_height - y_pad && j >= x_pad && j < input_image_width - x_pad)
            {
                int seg_index = i * input_image_width + j;
                cropped_seg[cropped_index] = seg_mask[seg_index];
                cropped_index++;
            }
        }
    }
    resizeSegMasks(cropped_seg, cropped_width, cropped_height, 1, seg_mask_real, ori_in_width, ori_in_height);
}

void YoloBasePostProcess::drawDetectionResults(cv::Mat &image, const ResultGroup &results) const
{
    // 定义类别颜色
    unsigned char class_colors[][3] = {
        {255, 56, 56},   // 'FF3838'
        {255, 157, 151}, // 'FF9D97'
        {255, 112, 31},  // 'FF701F'
        {255, 178, 29},  // 'FFB21D'
        {207, 210, 49},  // 'CFD231'
        {72, 249, 10},   // '48F90A'
        {146, 204, 23},  // '92CC17'
        {61, 219, 134},  // '3DDB86'
        {26, 147, 52},   // '1A9334'
        {0, 212, 187},   // '00D4BB'
        {44, 153, 168},  // '2C99A8'
        {0, 194, 255},   // '00C2FF'
        {52, 69, 147},   // '344593'
        {100, 115, 255}, // '6473FF'
        {0, 24, 236},    // '0018EC'
        {132, 56, 255},  // '8438FF'
        {82, 0, 133},    // '520085'
        {203, 56, 255},  // 'CB38FF'
        {255, 149, 200}, // 'FF95C8'
        {255, 55, 199}   // 'FF37C7'
    };

    int width = image.cols;
    int height = image.rows;
    float alpha = 0.5f; // 透明度

    // 首先绘制分割掩码
    if (results.count >= 1 && !results.segmentation_masks.empty())
    {
        // 直接修改原图的像素值
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                // 获取掩码值
                int mask_value = results.segmentation_masks[h * width + w];

                if (mask_value != 0)
                {
                    // 使用掩码值来索引颜色
                    cv::Vec3b color = cv::Vec3b(class_colors[mask_value % 20][0],
                                                class_colors[mask_value % 20][1],
                                                class_colors[mask_value % 20][2]); // RGB格式

                    cv::Vec3b &pixel = image.at<cv::Vec3b>(h, w);

                    // 使用对象的类别颜色来绘制掩码
                    pixel[0] = (unsigned char)(color[0] * (1 - alpha) + pixel[0] * alpha); // B
                    pixel[1] = (unsigned char)(color[1] * (1 - alpha) + pixel[1] * alpha); // G
                    pixel[2] = (unsigned char)(color[2] * (1 - alpha) + pixel[2] * alpha); // R
                }
            }
        }
    }

    // 然后绘制边界框和标签
    for (int i = 0; i < results.count; i++)
    {
        const DetectionObject *det_result = &results.results[i];

        // 获取对应类别的颜色
        cv::Scalar color = cv::Scalar(class_colors[det_result->cls_id % 20][2],
                                      class_colors[det_result->cls_id % 20][1],
                                      class_colors[det_result->cls_id % 20][0]); // BGR格式

        // 绘制边界框
        cv::rectangle(image,
                      cv::Point(det_result->box.left, det_result->box.top),
                      cv::Point(det_result->box.right, det_result->box.bottom),
                      color, 2);

        // 添加标签文本
        std::string label = "Class " + std::to_string(det_result->cls_id) + " " +
                            std::to_string(det_result->prop * 100) + "%";
        int baseline;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(image,
                      cv::Point(det_result->box.left, det_result->box.top - textSize.height - 10),
                      cv::Point(det_result->box.left + textSize.width, det_result->box.top),
                      color, -1);
        cv::putText(image, label,
                    cv::Point(det_result->box.left, det_result->box.top - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

} // namespace post_process
} // namespace deploy_percept