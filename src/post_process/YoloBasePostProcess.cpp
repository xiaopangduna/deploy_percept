#include "deploy_percept/post_process/YoloBasePostProcess.hpp"
#include "deploy_percept/types.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>
#include <vector>

namespace deploy_percept {
namespace post_process {

namespace
{

void resizeGray8Linear(
    const uint8_t *src,
    int src_w,
    int src_h,
    uint8_t *dst,
    int dst_w,
    int dst_h)
{
    if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0)
    {
        return;
    }

    if (src_w == dst_w && src_h == dst_h)
    {
        std::memcpy(dst, src, static_cast<std::size_t>(src_w) * static_cast<std::size_t>(src_h));
        return;
    }

    const float x_ratio = static_cast<float>(src_w) / static_cast<float>(dst_w);
    const float y_ratio = static_cast<float>(src_h) / static_cast<float>(dst_h);

    for (int y = 0; y < dst_h; ++y)
    {
        const float sy = (static_cast<float>(y) + 0.5f) * y_ratio - 0.5f;
        int y0 = static_cast<int>(std::floor(sy));
        int y1 = y0 + 1;
        const float wy = sy - static_cast<float>(y0);
        y0 = std::max(0, std::min(y0, src_h - 1));
        y1 = std::max(0, std::min(y1, src_h - 1));

        for (int x = 0; x < dst_w; ++x)
        {
            const float sx = (static_cast<float>(x) + 0.5f) * x_ratio - 0.5f;
            int x0 = static_cast<int>(std::floor(sx));
            int x1 = x0 + 1;
            const float wx = sx - static_cast<float>(x0);
            x0 = std::max(0, std::min(x0, src_w - 1));
            x1 = std::max(0, std::min(x1, src_w - 1));

            const float v00 = static_cast<float>(src[y0 * src_w + x0]);
            const float v01 = static_cast<float>(src[y0 * src_w + x1]);
            const float v10 = static_cast<float>(src[y1 * src_w + x0]);
            const float v11 = static_cast<float>(src[y1 * src_w + x1]);
            const float v0 = v00 + (v01 - v00) * wx;
            const float v1 = v10 + (v11 - v10) * wx;
            dst[y * dst_w + x] = static_cast<uint8_t>(std::lround(v0 + (v1 - v0) * wy));
        }
    }
}

} // namespace

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
        const uint8_t *src = &input_image[b * input_width * input_height];
        uint8_t *dst = &output_image[b * target_width * target_height];
        resizeGray8Linear(src, input_width, input_height, dst, target_width, target_height);
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

} // namespace post_process
} // namespace deploy_percept