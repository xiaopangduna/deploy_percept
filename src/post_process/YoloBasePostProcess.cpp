#include "deploy_percept/post_process/YoloBasePostProcess.hpp"
#include "deploy_percept/post_process/types.hpp"
#include <algorithm>
#include <set>
#include <vector>
#include <cmath>

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

int YoloBasePostProcess::nms(int validCount, std::vector<float>& outputLocations, 
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

} // namespace post_process
} // namespace deploy_percept
