#include "deploy_percept/engine/RknnEngine.hpp"
#include <vector>
#include <algorithm>
#include <set>
#include <cstring>

namespace deploy_percept
{
    namespace engine
    {

        RknnEngine::RknnEngine(const RknnEngine::Params &params)
            : params_(params)
        {
            
        }

        // bool RknnEngine::run(
        //     int8_t *input0,
        //     int8_t *input1,
        //     int8_t *input2,
        //     int model_in_h,
        //     int model_in_w,
        //     BoxRect pads,
        //     float scale_w,
        //     float scale_h,
        //     std::vector<int32_t> &qnt_zps,
        //     std::vector<float> &qnt_scales)
        // {
        //     // 重置结果

        //     return true;
        // }

    } // namespace engine
} // namespace deploy_percept