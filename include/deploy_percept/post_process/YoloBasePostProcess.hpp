#pragma once

#include "deploy_percept/post_process/BasePostProcess.hpp"
#include "deploy_percept/post_process/types.hpp"
#include <vector>
#include <string>

namespace deploy_percept {
namespace post_process {

class YoloBasePostProcess : public BasePostProcess {
public:
    YoloBasePostProcess();
    virtual ~YoloBasePostProcess() = default;

    // 通用的clamp函数 - 内联实现
    static inline int clamp(float val, int min, int max) {
        return val > min ? (val < max ? static_cast<int>(val) : max) : min;
    }

    // 通用的量化和反量化函数
    static int8_t qntF32ToAffine(float f32, int32_t zp, float scale);
    static float deqntAffineToF32(int8_t qnt, int32_t zp, float scale);
    static int32_t clip(float val, float min, float max);

    // 通用的NMS实现
    static int nms(int validCount, std::vector<float>& outputLocations, std::vector<int> classIds, 
            std::vector<int>& order, int filterId, float threshold);

    // 通用的边界框重叠计算
    static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, 
                                float xmin1, float ymin1, float xmax1, float ymax1);
};

} // namespace post_process
} // namespace deploy_percept

