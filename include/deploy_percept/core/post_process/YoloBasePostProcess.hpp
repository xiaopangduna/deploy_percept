#pragma once

#include "deploy_percept/core/post_process/BasePostProcess.hpp"
#include "deploy_percept/common/types.hpp"

#include <cstdint>
#include <vector>

namespace deploy_percept
{
    namespace post_process
    {

        /** YOLO 后处理共用 static 工具（NMS、量化、sigmoid、分割 mask 等），无实例状态。 */
        class YoloBasePostProcess : public BasePostProcess
        {
        public:
            /** YOLOv5 COCO 默认 anchor（stride 8/16/32 各 3 组 wh） */
            struct YoloV5CocoAnchors
            {
                static inline const std::vector<int> kStride8{10, 13, 16, 30, 33, 23};
                static inline const std::vector<int> kStride16{30, 61, 62, 45, 59, 119};
                static inline const std::vector<int> kStride32{116, 90, 156, 198, 373, 326};
            };

            static float sigmoid(float x);
            static float desigmoid(float x);

            static int clamp(float val, int min, int max);
            static int8_t qntF32ToAffine(float f32, int32_t zp, float scale);
            static float deqntAffineToF32(int8_t qnt, int32_t zp, float scale);
            static int32_t clip(float val, float min, float max);

            /** DFL 解码：tensor 为 4×dfl_len 分布，输出 box[4] 为距离偏移 */
            static void decodeDflBox(const float *tensor, int dfl_len, float box[4]);

            /** 对指定类别的候选框做 NMS；order 中 -1 表示被抑制；box_stride 为每框字段数（默认 4，Pose 为 5） */
            static int retainHighestScoringBoxesByNMS(
                int validCount,
                std::vector<float> &outputLocations,
                std::vector<int> classIds,
                std::vector<int> &order,
                int filterId,
                float threshold,
                int box_stride = 4);

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

            /**
             * Seg 共用：由 NMS 后 detections + seg coeff + proto 合成 label_mask。
             * workspace 向量由调用方持有并传入，避免重复分配。
             */
            static void buildLabelMask(
                const std::vector<Detection> &detections,
                const std::vector<float> &seg_coeffs,
                float *proto,
                int proto_channel,
                int proto_height,
                int proto_width,
                int image_width,
                int image_height,
                std::vector<std::uint8_t> &label_mask,
                std::vector<std::uint8_t> &matmul_out,
                std::vector<std::uint8_t> &seg_mask,
                std::vector<std::uint8_t> &all_mask_in_one);
        };

    } // namespace post_process
} // namespace deploy_percept
