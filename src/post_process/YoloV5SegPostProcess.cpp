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
            result_.group.results_seg.resize(1); // 只需要一个分割结果
        }

        // 绘制检测和分割结果
        void YoloV5SegPostProcess::drawDetectionResults(cv::Mat &image, const ResultGroup &results) const
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
            if (results.count >= 1)
            {
                for (int i = 0; i < results.count; i++)
                {
                    const DetectResult *det_result = &results.results[i];

                    // 获取对应类别的颜色
                    cv::Vec3b color = cv::Vec3b(class_colors[det_result->cls_id % 20][0],
                                                class_colors[det_result->cls_id % 20][1],
                                                class_colors[det_result->cls_id % 20][2]); // RGB格式

                    // 绘制分割掩码
                    if (results.results_seg.size() > i && results.results_seg[i].seg_mask != nullptr)
                    {
                        // 直接修改原图的像素值
                        for (int h = 0; h < height; h++)
                        {
                            for (int w = 0; w < width; w++)
                            {
                                // 获取掩码值
                                int mask_value = results.results_seg[i].seg_mask[h * width + w];

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
                }

                // 然后绘制边界框和标签
                for (int i = 0; i < results.count; i++)
                {
                    const DetectResult *det_result = &results.results[i];

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
        }

        // 删除重复的quick_sort_indice_inverse函数，使用父类的quickSortIndices实现

        int YoloV5SegPostProcess::process_i8(std::vector<void *> *all_input, int input_id, int *anchor, int grid_h, int grid_w,
                                             int stride,
                                             std::vector<float> &boxes, std::vector<float> &segments,
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

            // 原型掩码 (proto) 现由 run() 函数单独处理

            int8_t *input = (int8_t *)(*all_input)[input_id];
            int8_t *input_seg = (int8_t *)(*all_input)[input_id + 1];
            int32_t zp = output_zps[input_id];
            float scale = output_scales[input_id];
            int32_t zp_seg = output_zps[input_id + 1];
            float scale_seg = output_scales[input_id + 1];

            int8_t thres_i8 = qntF32ToAffine(threshold, zp, scale);

            for (int a = 0; a < 3; a++)
            {
                for (int i = 0; i < grid_h; i++)
                {
                    for (int j = 0; j < grid_w; j++)
                    {
                        int8_t box_confidence = input[((params_.prop_box_size + params_.obj_class_num) * a + 4) * grid_len + i * grid_w + j];
                        if (box_confidence >= thres_i8)
                        {
                            int offset = ((params_.prop_box_size + params_.obj_class_num) * a) * grid_len + i * grid_w + j;
                            int offset_seg = (params_.proto_channel * a) * grid_len + i * grid_w + j;
                            int8_t *in_ptr = input + offset;
                            int8_t *in_ptr_seg = input_seg + offset_seg;

                            float box_conf_f32 = deqntAffineToF32(box_confidence, zp, scale);

                            if (box_conf_f32 > threshold)
                            {
                                // 在条件内部进行类别概率计算和边界框解码
                                int8_t maxClassProbs = in_ptr[5 * grid_len];
                                int maxClassId = 0;
                                for (int k = 1; k < params_.obj_class_num; ++k)
                                {
                                    int8_t prob = in_ptr[(5 + k) * grid_len];
                                    if (prob > maxClassProbs)
                                    {
                                        maxClassId = k;
                                        maxClassProbs = prob;
                                    }
                                }

                                float class_prob_f32 = deqntAffineToF32(maxClassProbs, zp, scale);
                                float limit_score = box_conf_f32 * class_prob_f32;

                                if (limit_score > threshold)
                                {
                                    // 边界框解码
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

                                    for (int k = 0; k < params_.proto_channel; k++)
                                    {
                                        float seg_element_fp = deqntAffineToF32(in_ptr_seg[(k)*grid_len], zp_seg, scale_seg);
                                        segments.push_back(seg_element_fp);
                                    }

                                    objProbs.push_back(limit_score);
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
            }
            return validCount;
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
            resize_by_opencv_uint8(cropped_seg, cropped_width, cropped_height, 1, seg_mask_real, ori_in_width, ori_in_height);
        }

        void YoloV5SegPostProcess::processNMSSelectedResults(
            const std::vector<int> &indexArray,
            const std::vector<float> &filterBoxes,
            const std::vector<int> &classId,
            const std::vector<float> &objProbs,
            const std::vector<float> &filterSegments,
            int validCount,
            std::vector<float> &filterSegments_by_nms,
            int &last_count)
        {
            for (int i = 0; i < validCount; ++i)
            {
                // 跳过被NMS标记为冗余的检测框，或超出最大检测数量限制的框
                if (indexArray[i] == -1 || last_count >= params_.obj_numb_max_size)
                {
                    continue;
                }

                int n = indexArray[i]; // 获取原始检测框的索引

                // 提取并计算边界框坐标
                float x1 = filterBoxes[n * 4 + 0];      // 左上角x
                float y1 = filterBoxes[n * 4 + 1];      // 左上角y
                float x2 = x1 + filterBoxes[n * 4 + 2]; // 右下角x (x1 + width)
                float y2 = y1 + filterBoxes[n * 4 + 3]; // 右下角y (y1 + height)

                int id = classId[n];          // 保存真实的类别ID
                float obj_conf = objProbs[i]; // 获取该检测框的置信度

                // 收集该检测框对应的分割特征向量
                for (int k = 0; k < params_.proto_channel; k++)
                {
                    filterSegments_by_nms.push_back(filterSegments[n * params_.proto_channel + k]);
                }

                // 填充检测结果结构体
                result_.group.results[last_count].box.left = x1;
                result_.group.results[last_count].box.top = y1;
                result_.group.results[last_count].box.right = x2;
                result_.group.results[last_count].box.bottom = y2;

                result_.group.results[last_count].prop = obj_conf;

                // 设置类别名称
                snprintf(result_.group.results[last_count].name,
                         sizeof(result_.group.results[last_count].name),
                         "class_%d", id);
                
                // 设置类别ID
                result_.group.results[last_count].cls_id = id;

                last_count++; // 增加有效检测计数
            }
        }

        bool YoloV5SegPostProcess::run(
            std::vector<std::vector<int>> &output_dims,
            std::vector<float> &output_scales,
            std::vector<int32_t> &output_zps,
            std::vector<void *> *outputs,
            int input_image_width,
            int input_image_height)
        {
            // Print debug info to understand dimensions
            printf("Processing image: %dx%d\n", input_image_width, input_image_height);

            std::vector<float> filterBoxes;
            std::vector<float> objProbs;
            std::vector<int> classId;

            std::vector<float> filterSegments;
            float proto[params_.proto_channel * params_.proto_height * params_.proto_weight];
            std::vector<float> filterSegments_by_nms;

            int validCount = 0;
            int stride = 0;
            int grid_h = 0;
            int grid_w = 0;

            // Initialize result
            result_.group.count = 0;
            result_.success = false;

            // 单独处理原型掩码层（Layer 6）
            if (output_dims.size() > 6)
            {
                int8_t *input_proto = (int8_t *)(*outputs)[6];
                int32_t zp_proto = output_zps[6];
                float scale_proto = output_scales[6];

                for (int i = 0; i < params_.proto_channel * params_.proto_height * params_.proto_weight; i++)
                {
                    proto[i] = deqntAffineToF32(input_proto[i], zp_proto, scale_proto);
                }
            }

            // Process the outputs of the model (only layers 0-5)
            for (int i = 0; i < 6; i++) // 只处理前6层
            {
                if (i >= output_dims.size())
                    break;
                grid_h = output_dims[i][2];
                grid_w = output_dims[i][3];
                stride = input_image_height / grid_h;

                // 根据层索引选择对应的anchor
                const int *current_anchor;
                switch (i / 2)
                {
                case 0:
                    current_anchor = params_.anchor_stride8.data();
                    break;
                case 1:
                    current_anchor = params_.anchor_stride16.data();
                    break;
                case 2:
                    current_anchor = params_.anchor_stride32.data();
                    break;
                default:
                    current_anchor = params_.anchor_stride8.data();
                    break;
                }

                validCount += process_i8(outputs, i, (int *)current_anchor, grid_h, grid_w,
                                         stride,
                                         filterBoxes, filterSegments, objProbs, classId, params_.conf_threshold,
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

            // 明确调用基类的静态函数
            YoloBasePostProcess::quickSortIndices(objProbs, 0, validCount - 1, indexArray);

            std::set<int> class_set(std::begin(classId), std::end(classId));

            for (auto c : class_set)
            {
                nms(validCount, filterBoxes, classId, indexArray, c, params_.nms_threshold);
            }

            int last_count = 0;
            result_.group.count = 0;

            // 处理NMS筛选后的检测结果
            processNMSSelectedResults(indexArray, filterBoxes, classId, objProbs,
                                      filterSegments, validCount, filterSegments_by_nms, last_count);

            result_.group.count = last_count;
            int boxes_num = result_.group.count;

            // 如果没有检测到物体，不需要生成分割掩码
            if (boxes_num <= 0)
            {
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
                if (pos != std::string::npos)
                {
                    cls_id[i] = std::stoi(name_str.substr(pos + 1));
                }
                else
                {
                    cls_id[i] = 0; // Default to 0 if parsing fails
                }
            }

            // compute the mask through Matmul
            int ROWS_A = boxes_num;
            int COLS_A = params_.proto_channel;
            int COLS_B = params_.proto_height * params_.proto_weight;
            uint8_t *matmul_out = nullptr;

            // Allocate memory for matmul result
            matmul_out = (uint8_t *)malloc(boxes_num * params_.proto_height * params_.proto_weight * sizeof(uint8_t));
            if (!matmul_out)
            {
                result_.message = "Failed to allocate memory for matmul_out";
                return false;
            }

            // Perform matrix multiplication: instance coefficients * prototype masks
            matmul_by_cpu_uint8(filterSegments_by_nms, proto, matmul_out, boxes_num, params_.proto_channel, params_.proto_height * params_.proto_weight);

            // Resize the matmul result to model resolution
            uint8_t *seg_mask = (uint8_t *)malloc(boxes_num * input_image_height * input_image_width * sizeof(uint8_t));
            if (!seg_mask)
            {
                free(matmul_out);
                result_.message = "Failed to allocate memory for seg_mask";
                return false;
            }

            resize_by_opencv_uint8(matmul_out, params_.proto_weight, params_.proto_height, boxes_num, seg_mask, input_image_width, input_image_height);

            // Allocate memory for combined mask
            uint8_t *all_mask_in_one = (uint8_t *)malloc(input_image_height * input_image_width * sizeof(uint8_t));
            if (!all_mask_in_one)
            {
                free(seg_mask);
                free(matmul_out);
                result_.message = "Failed to allocate memory for combined mask";
                return false;
            }

            // Initialize combined mask to 0
            memset(all_mask_in_one, 0, input_image_height * input_image_width * sizeof(uint8_t));

            // Crop mask based on bounding boxes
            crop_mask_uint8(seg_mask, all_mask_in_one, filterBoxes_by_nms, boxes_num, cls_id, input_image_height, input_image_width);

            // get real mask - simplified without pad handling
            uint8_t *real_seg_mask = (uint8_t *)malloc(input_image_height * input_image_width * sizeof(uint8_t));
            if (!real_seg_mask)
            {
                free(all_mask_in_one);
                free(seg_mask);
                free(matmul_out);
                result_.message = "Failed to allocate memory for real_seg_mask";
                return false;
            }

            memcpy(real_seg_mask, all_mask_in_one, input_image_height * input_image_width * sizeof(uint8_t));

            result_.group.results_seg[0].seg_mask = real_seg_mask;

            // Clean up allocated memory
            free(all_mask_in_one);
            free(seg_mask);
            free(matmul_out);

            result_.success = true;
            return true;
        }
    } // namespace post_process
} // namespace deploy_percept
