#include "deploy_percept/post_process/YoloV5SegPostProcess.hpp"
#include <set>
#include <algorithm>
#include <cmath>
#include <malloc.h>

namespace deploy_percept
{
    namespace post_process
    {
        // 构造函数
        YoloV5SegPostProcess::YoloV5SegPostProcess(const Params &params) : params_(params)
        {
            result_.group.results.resize(params_.obj_numb_max_size);
            result_.group.results_seg.resize(params_.obj_numb_max_size);
        }

        
        inline static int32_t __clip(float val, float min, float max)
        {
            float f = val <= min ? min : (val >= max ? max : val);
            return f;
        }
        
        static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
        {
            float dst_val = (f32 / scale) + zp;
            int8_t res = (int8_t)__clip(dst_val, -128, 127);
            return res;
        }
        
        // Anchor values for YOLOv5
        const int anchor[3][6] = {{10, 13, 16, 30, 33, 23},
                                  {30, 61, 62, 45, 59, 119},
                                  {116, 90, 156, 198, 373, 326}};

        int YoloV5SegPostProcess::process_i8(std::vector<void*> *all_input, int input_id, int *anchor, int grid_h, int grid_w, 
                                           int height, int width, int stride,
                                           std::vector<float> &boxes, std::vector<float> &segments, float *proto, 
                                           std::vector<float> &objProbs, std::vector<int> &classId, float threshold,
                                           std::vector<std::vector<int>> &output_dims, std::vector<float> &output_scales, 
                                           std::vector<int32_t> &output_zps)
        {
            int validCount = 0;
            int grid_len = grid_h * grid_w;

            if (input_id % 2 == 1)
            {
                return validCount;
            }

            const int PROTO_CHANNEL = 32;
            const int PROTO_HEIGHT = 160;
            const int PROTO_WEIGHT = 160;
            const int OBJ_CLASS_NUM = 80;
            const int PROP_BOX_SIZE = (5 + OBJ_CLASS_NUM);

            if (input_id == 6)
            {
                int8_t *input_proto = (int8_t *)(*all_input)[input_id];
                int32_t zp_proto = output_zps[input_id];
                float scale_proto = output_scales[input_id];
                for (int i = 0; i < PROTO_CHANNEL * PROTO_HEIGHT * PROTO_WEIGHT; i++)
                {
                    proto[i] = deqntAffineToF32(input_proto[i], zp_proto, scale_proto);
                }
                return validCount;
            }

            int8_t *input = (int8_t *)(*all_input)[input_id];
            int8_t *input_seg = (int8_t *)(*all_input)[input_id + 1];
            int32_t zp = output_zps[input_id];
            float scale = output_scales[input_id];
            int32_t zp_seg = output_zps[input_id + 1];
            float scale_seg = output_scales[input_id + 1];

            int8_t thres_i8 = qnt_f32_to_affine(threshold, zp, scale);

            for (int a = 0; a < 3; a++)
            {
                for (int i = 0; i < grid_h; i++)
                {
                    for (int j = 0; j < grid_w; j++)
                    {
                        int8_t box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
                        if (box_confidence >= thres_i8)
                        {
                            int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
                            int offset_seg = (PROTO_CHANNEL * a) * grid_len + i * grid_w + j;
                            int8_t *in_ptr = input + offset;
                            int8_t *in_ptr_seg = input_seg + offset_seg;

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
                            for (int k = 1; k < OBJ_CLASS_NUM; ++k)
                            {
                                int8_t prob = in_ptr[(5 + k) * grid_len];
                                if (prob > maxClassProbs)
                                {
                                    maxClassId = k;
                                    maxClassProbs = prob;
                                }
                            }

                            float box_conf_f32 = deqntAffineToF32(box_confidence, zp, scale);
                            float class_prob_f32 = deqntAffineToF32(maxClassProbs, zp, scale);
                            float limit_score = box_conf_f32 * class_prob_f32;
                            if (limit_score > threshold)
                            {
                                for (int k = 0; k < PROTO_CHANNEL; k++)
                                {
                                    float seg_element_fp = deqntAffineToF32(in_ptr_seg[(k)*grid_len], zp_seg, scale_seg);
                                    segments.push_back(seg_element_fp);
                                }

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

        int YoloV5SegPostProcess::quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices)
        {
            float key;
            int key_index;
            int low = left;
            int high = right;
            if (left < right)
            {
                key_index = indices[left];
                key = input[left];
                while (low < high)
                {
                    while (low < high && input[high] <= key)
                    {
                        high--;
                    }
                    input[low] = input[high];
                    indices[low] = indices[high];
                    while (low < high && input[low] >= key)
                    {
                        low++;
                    }
                    input[high] = input[low];
                    indices[high] = indices[low];
                }
                input[low] = key;
                indices[low] = key_index;
                quick_sort_indice_inverse(input, left, low - 1, indices);
                quick_sort_indice_inverse(input, low + 1, right, indices);
            }
            return low;
        }



        int YoloV5SegPostProcess::clamp(float val, int min, int max)
        {
            return val > min ? (val < max ? val : max) : min;
        }

        int YoloV5SegPostProcess::box_reverse(int position, int boundary, int pad, float scale)
        {
            return (int)((clamp(position, 0, boundary) - pad) / scale);
        }

        void YoloV5SegPostProcess::matmul_by_cpu_uint8(std::vector<float> &A, float *B, uint8_t *C, int ROWS_A, int COLS_A, int COLS_B)
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

        void YoloV5SegPostProcess::resize_by_opencv_uint8(uint8_t *input_image, int input_width, int input_height, int boxes_num, 
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

        void YoloV5SegPostProcess::crop_mask_uint8(uint8_t *seg_mask, uint8_t *all_mask_in_one, float *boxes, int boxes_num, 
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

        void YoloV5SegPostProcess::seg_reverse(uint8_t *seg_mask, uint8_t *cropped_seg, uint8_t *seg_mask_real,
                                             int model_in_height, int model_in_width, int cropped_height, int cropped_width, 
                                             int ori_in_height, int ori_in_width, int y_pad, int x_pad)
        {
            if (y_pad == 0 && x_pad == 0 && ori_in_height == model_in_height && ori_in_width == model_in_width)
            {
                memcpy(seg_mask_real, seg_mask, ori_in_height * ori_in_width);
                return;
            }

            int cropped_index = 0;
            for (int i = 0; i < model_in_height; i++)
            {
                for (int j = 0; j < model_in_width; j++)
                {
                    if (i >= y_pad && i < model_in_height - y_pad && j >= x_pad && j < model_in_width - x_pad)
                    {
                        int seg_index = i * model_in_width + j;
                        cropped_seg[cropped_index] = seg_mask[seg_index];
                        cropped_index++;
                    }
                }
            }
            // Note: Here are different methods provided for implementing single-channel image scaling.
            //       The method of using rga to resize the image requires that the image size is 2 aligned.
            resize_by_opencv_uint8(cropped_seg, cropped_width, cropped_height, 1, seg_mask_real, ori_in_width, ori_in_height);
            // resize_by_rga_rk356x(cropped_seg, cropped_width, cropped_height, seg_mask_real, ori_in_width, ori_in_height);
            // resize_by_rga_rk3588(cropped_seg, cropped_width, cropped_height, seg_mask_real, ori_in_width, ori_in_height);
        }

        bool YoloV5SegPostProcess::run(
            int model_in_width,
            int model_in_height,
            std::vector<std::vector<int>> &output_dims,
            std::vector<float> &output_scales,
            std::vector<int32_t> &output_zps,
            std::vector<void*> *outputs,
            BoxRect pads,
            float scale,
            int input_image_width,
            int input_image_height)
        {
            const int PROTO_CHANNEL = params_.proto_channel;
            const int PROTO_HEIGHT = params_.proto_height;
            const int PROTO_WEIGHT = params_.proto_weight;
            const int OBJ_CLASS_NUM = params_.obj_class_num;
            const int OBJ_NUMB_MAX_SIZE = params_.obj_numb_max_size;
            const int PROP_BOX_SIZE = (5 + OBJ_CLASS_NUM);

            std::vector<float> filterBoxes;
            std::vector<float> objProbs;
            std::vector<int> classId;

            std::vector<float> filterSegments;
            float proto[PROTO_CHANNEL * PROTO_HEIGHT * PROTO_WEIGHT];
            std::vector<float> filterSegments_by_nms;

            int validCount = 0;
            int stride = 0;
            int grid_h = 0;
            int grid_w = 0;

            // Initialize result
            result_.group.count = 0;
            result_.success = false;

            // Process the outputs of the model
            for (int i = 0; i < 7; i++)
            {
                if (i >= output_dims.size()) break;
                grid_h = output_dims[i][2];
                grid_w = output_dims[i][3];
                stride = model_in_height / grid_h;
                validCount += process_i8(outputs, i, (int *)anchor[i / 2], grid_h, grid_w, model_in_height, model_in_width, stride, 
                                         filterBoxes, filterSegments, proto, objProbs, classId, params_.conf_threshold, 
                                         output_dims, output_scales, output_zps);
            }

            // NMS
            if (validCount <= 0)
            {
                result_.success = true;
                return true;
            }
            
            std::vector<int> indexArray;
            for (int i = 0; i < validCount; ++i)
            {
                indexArray.push_back(i);
            }

            quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

            std::set<int> class_set(std::begin(classId), std::end(classId));

            for (auto c : class_set)
            {
                nms(validCount, filterBoxes, classId, indexArray, c, params_.nms_threshold);
            }

            int last_count = 0;
            result_.group.count = 0;

            // Resize vectors to ensure they have enough space
            result_.group.results.resize(OBJ_NUMB_MAX_SIZE);
            result_.group.results_seg.resize(1); // 只需要一个分割结果

            for (int i = 0; i < validCount; ++i)
            {
                if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE)
                {
                    continue;
                }
                int n = indexArray[i];

                float x1 = filterBoxes[n * 4 + 0];
                float y1 = filterBoxes[n * 4 + 1];
                float x2 = x1 + filterBoxes[n * 4 + 2];
                float y2 = y1 + filterBoxes[n * 4 + 3];
                int id = classId[n];  // 保存真实的类别ID
                float obj_conf = objProbs[i]; // 修复：使用正确的索引获取置信度

                for (int k = 0; k < PROTO_CHANNEL; k++)
                {
                    filterSegments_by_nms.push_back(filterSegments[n * PROTO_CHANNEL + k]);
                }

                result_.group.results[last_count].box.left = x1;
                result_.group.results[last_count].box.top = y1;
                result_.group.results[last_count].box.right = x2;
                result_.group.results[last_count].box.bottom = y2;

                result_.group.results[last_count].prop = obj_conf;
                // Set class name based on class id
                snprintf(result_.group.results[last_count].name, 
                         sizeof(result_.group.results[last_count].name), 
                         "class_%d", id);
                // 保存真实的类别ID
                // 注意：在DetectResult结构中没有cls_id字段，所以我们需要在生成掩码时使用正确的类别ID

                last_count++;
            }
            result_.group.count = last_count;
            int boxes_num = result_.group.count;

            // 如果没有检测到物体，不需要生成分割掩码
            if (boxes_num <= 0) {
                result_.success = true;
                return true;
            }

            float filterBoxes_by_nms[boxes_num * 4];
            int cls_id[boxes_num];
            for (int i = 0; i < boxes_num; i++)
            {
                // for crop_mask
                filterBoxes_by_nms[i * 4 + 0] = result_.group.results[i].box.left;   // x1;
                filterBoxes_by_nms[i * 4 + 1] = result_.group.results[i].box.top;    // y1;
                filterBoxes_by_nms[i * 4 + 2] = result_.group.results[i].box.right;  // x2;
                filterBoxes_by_nms[i * 4 + 3] = result_.group.results[i].box.bottom; // y2;
                
                // 获取真实的类别ID
                std::string name_str(result_.group.results[i].name);
                size_t pos = name_str.find_last_of('_');
                if (pos != std::string::npos) {
                    cls_id[i] = std::stoi(name_str.substr(pos + 1));
                } else {
                    cls_id[i] = 0;  // Default to 0 if parsing fails
                }

                // get real box
                result_.group.results[i].box.left = box_reverse(result_.group.results[i].box.left, model_in_width, pads.left, scale);
                result_.group.results[i].box.top = box_reverse(result_.group.results[i].box.top, model_in_height, pads.top, scale);
                result_.group.results[i].box.right = box_reverse(result_.group.results[i].box.right, model_in_width, pads.left, scale);
                result_.group.results[i].box.bottom = box_reverse(result_.group.results[i].box.bottom, model_in_height, pads.top, scale);
            }

            // compute the mask through Matmul
            int ROWS_A = boxes_num;
            int COLS_A = PROTO_CHANNEL;
            int COLS_B = PROTO_HEIGHT * PROTO_WEIGHT;
            uint8_t *matmul_out = nullptr;
            
            // Allocate memory for matmul result
            matmul_out = (uint8_t *)malloc(boxes_num * PROTO_HEIGHT * PROTO_WEIGHT * sizeof(uint8_t));
            if (!matmul_out) {
                result_.message = "Failed to allocate memory for matmul_out";
                return false;
            }
            
            // Perform matrix multiplication: instance coefficients * prototype masks
            matmul_by_cpu_uint8(filterSegments_by_nms, proto, matmul_out, ROWS_A, COLS_A, COLS_B);

            // Resize the matmul result to model resolution
            uint8_t *seg_mask = (uint8_t *)malloc(boxes_num * model_in_height * model_in_width * sizeof(uint8_t));
            if (!seg_mask) {
                free(matmul_out);
                result_.message = "Failed to allocate memory for seg_mask";
                return false;
            }
            
            resize_by_opencv_uint8(matmul_out, PROTO_WEIGHT, PROTO_HEIGHT, boxes_num, seg_mask, model_in_width, model_in_height);

            // Allocate memory for combined mask
            uint8_t *all_mask_in_one = (uint8_t *)malloc(model_in_height * model_in_width * sizeof(uint8_t));
            if (!all_mask_in_one) {
                free(seg_mask);
                free(matmul_out);
                result_.message = "Failed to allocate memory for combined mask";
                return false;
            }
            
            // Initialize combined mask to 0
            memset(all_mask_in_one, 0, model_in_height * model_in_width * sizeof(uint8_t));

            // Crop mask based on bounding boxes
            crop_mask_uint8(seg_mask, all_mask_in_one, filterBoxes_by_nms, boxes_num, cls_id, model_in_height, model_in_width);

            // get real mask
            int cropped_height = model_in_height - pads.top * 2;  // Assuming symmetric padding
            int cropped_width = model_in_width - pads.left * 2;   // Assuming symmetric padding
            int ori_in_height = input_image_height;
            int ori_in_width = input_image_width;
            int y_pad = pads.top;
            int x_pad = pads.left;
            
            uint8_t *cropped_seg_mask = (uint8_t *)malloc(cropped_height * cropped_width * sizeof(uint8_t));
            if (!cropped_seg_mask) {
                free(all_mask_in_one);
                free(seg_mask);
                free(matmul_out);
                result_.message = "Failed to allocate memory for cropped_seg_mask";
                return false;
            }
            
            uint8_t *real_seg_mask = (uint8_t *)malloc(ori_in_height * ori_in_width * sizeof(uint8_t));
            if (!real_seg_mask) {
                free(cropped_seg_mask);
                free(all_mask_in_one);
                free(seg_mask);
                free(matmul_out);
                result_.message = "Failed to allocate memory for real_seg_mask";
                return false;
            }
            
            seg_reverse(all_mask_in_one, cropped_seg_mask, real_seg_mask,
                       model_in_height, model_in_width, cropped_height, cropped_width, 
                       ori_in_height, ori_in_width, y_pad, x_pad);
            
            result_.group.results_seg[0].seg_mask = real_seg_mask;
            
            // Clean up allocated memory
            free(cropped_seg_mask);
            free(all_mask_in_one);
            free(seg_mask);
            free(matmul_out);

            result_.success = true;
            return true;
        }
    } // namespace post_process
} // namespace deploy_percept