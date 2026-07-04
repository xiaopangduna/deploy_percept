#pragma once

#include "deploy_percept/core/post_process/BasePostProcess.hpp"
#include "deploy_percept/common/types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace deploy_percept
{
    namespace post_process
    {

        /** YOLO 后处理共用 static 工具（NMS、量化、sigmoid、分割 mask 等），无实例状态。 */
        class YoloBasePostProcess : public BasePostProcess
        {
        public:
            YoloBasePostProcess() = default;
            virtual ~YoloBasePostProcess() = default;

            static float sigmoid(float x);
            static float desigmoid(float x);

            static int clamp(float val, int min, int max);
            static int8_t qntF32ToAffine(float f32, int32_t zp, float scale);
            static float deqntAffineToF32(int8_t qnt, int32_t zp, float scale);
            static int32_t clip(float val, float min, float max);

            /** 按 scores 降序快速排序 indices（NMS 前使用） */
            static int quickSortIndices(
                std::vector<float> &input,
                int left,
                int right,
                std::vector<int> &indices);

            /** 对指定类别的候选框做 NMS；order 中 -1 表示被抑制 */
            static int retainHighestScoringBoxesByNMS(
                int validCount,
                std::vector<float> &outputLocations,
                std::vector<int> classIds,
                std::vector<int> &order,
                int filterId,
                float threshold);

            static float CalculateOverlap(
                float xmin0,
                float ymin0,
                float xmax0,
                float ymax0,
                float xmin1,
                float ymin1,
                float xmax1,
                float ymax1);

            static void computeSegMask(
                std::vector<float> &A,
                float *B,
                uint8_t *C,
                int ROWS_A,
                int COLS_A,
                int COLS_B);

            static void resizeSegMasks(
                uint8_t *input_image,
                int input_width,
                int input_height,
                int boxes_num,
                uint8_t *output_image,
                int target_width,
                int target_height);

            static void mergeBoxMasks(
                uint8_t *seg_mask,
                uint8_t *all_mask_in_one,
                float *boxes,
                int boxes_num,
                int *cls_id,
                int height,
                int width);
        };

    } // namespace post_process
} // namespace deploy_percept
