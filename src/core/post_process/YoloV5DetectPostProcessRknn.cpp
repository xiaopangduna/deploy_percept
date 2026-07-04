#include "deploy_percept/core/post_process/YoloV5DetectPostProcessRknn.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace deploy_percept
{
    namespace post_process
    {

        YoloV5DetectPostProcessRknn::YoloV5DetectPostProcessRknn(const Params &params)
            : params_(params)
        {
        }

        bool YoloV5DetectPostProcessRknn::run(
            const std::vector<int8_t *> &inputs,
            int model_in_h,
            int model_in_w,
            std::vector<int32_t> &qnt_zps,
            std::vector<float> &qnt_scales)
        {
            if (inputs.size() != 3)
            {
                result_.success = false;
                result_.message =
                    "Invalid input count: expected 3 inputs, got " + std::to_string(inputs.size());
                return false;
            }

            resetResult(result_);

            std::vector<float> filter_boxes;
            std::vector<float> obj_probs;
            std::vector<int> class_ids;

            const int valid_count0 = decodeDetectionHead(
                inputs[0],
                params_.anchor_stride8.data(),
                model_in_h / 8,
                model_in_w / 8,
                8,
                filter_boxes,
                obj_probs,
                class_ids,
                params_.conf_threshold,
                qnt_zps[0],
                qnt_scales[0]);

            const int valid_count1 = decodeDetectionHead(
                inputs[1],
                params_.anchor_stride16.data(),
                model_in_h / 16,
                model_in_w / 16,
                16,
                filter_boxes,
                obj_probs,
                class_ids,
                params_.conf_threshold,
                qnt_zps[1],
                qnt_scales[1]);

            const int valid_count2 = decodeDetectionHead(
                inputs[2],
                params_.anchor_stride32.data(),
                model_in_h / 32,
                model_in_w / 32,
                32,
                filter_boxes,
                obj_probs,
                class_ids,
                params_.conf_threshold,
                qnt_zps[2],
                qnt_scales[2]);

            const int valid_count = valid_count0 + valid_count1 + valid_count2;
            if (valid_count <= 0)
            {
                result_.message = "No objects detected";
                return false;
            }

            std::vector<int> index_array;
            applyNmsToCandidates(
                filter_boxes, obj_probs, class_ids, valid_count, params_.nms_threshold, index_array);
            return collectDetections(
                index_array,
                filter_boxes,
                obj_probs,
                class_ids,
                valid_count,
                params_.obj_numb_max_size,
                params_.obj_name_max_size,
                result_,
                model_in_w,
                model_in_h);
        }

        int YoloV5DetectPostProcessRknn::decodeDetectionHead(
            int8_t *input,
            int *anchor,
            int grid_h,
            int grid_w,
            int stride,
            std::vector<float> &boxes,
            std::vector<float> &obj_probs,
            std::vector<int> &class_ids,
            float threshold,
            int32_t zp,
            float scale)
        {
            int valid_count = 0;
            const int grid_len = grid_h * grid_w;
            const int8_t thres_i8 = qntF32ToAffine(threshold, zp, scale);
            const int prop_box_size = (5 + params_.obj_class_num);

            for (int a = 0; a < 3; a++)
            {
                for (int i = 0; i < grid_h; i++)
                {
                    for (int j = 0; j < grid_w; j++)
                    {
                        const int8_t box_confidence =
                            input[(prop_box_size * a + 4) * grid_len + i * grid_w + j];
                        if (box_confidence >= thres_i8)
                        {
                            const int offset = (prop_box_size * a) * grid_len + i * grid_w + j;
                            int8_t *in_ptr = input + offset;

                            float box_x = (deqntAffineToF32(*in_ptr, zp, scale)) * 2.0 - 0.5;
                            float box_y = (deqntAffineToF32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
                            float box_w = (deqntAffineToF32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
                            float box_h = (deqntAffineToF32(in_ptr[3 * grid_len], zp, scale)) * 2.0;

                            box_x = (box_x + j) * static_cast<float>(stride);
                            box_y = (box_y + i) * static_cast<float>(stride);

                            box_w = box_w * box_w * static_cast<float>(anchor[a * 2]);
                            box_h = box_h * box_h * static_cast<float>(anchor[a * 2 + 1]);

                            box_x -= (box_w / 2.0);
                            box_y -= (box_h / 2.0);

                            int8_t max_class_probs = in_ptr[5 * grid_len];
                            int max_class_id = 0;
                            for (int k = 1; k < params_.obj_class_num; ++k)
                            {
                                const int8_t prob = in_ptr[(5 + k) * grid_len];
                                if (prob > max_class_probs)
                                {
                                    max_class_id = k;
                                    max_class_probs = prob;
                                }
                            }

                            if (max_class_probs > thres_i8)
                            {
                                obj_probs.push_back(
                                    deqntAffineToF32(max_class_probs, zp, scale) *
                                    deqntAffineToF32(box_confidence, zp, scale));
                                class_ids.push_back(max_class_id);
                                ++valid_count;

                                boxes.push_back(box_x);
                                boxes.push_back(box_y);
                                boxes.push_back(box_w);
                                boxes.push_back(box_h);
                            }
                        }
                    }
                }
            }
            return valid_count;
        }

    } // namespace post_process
} // namespace deploy_percept
