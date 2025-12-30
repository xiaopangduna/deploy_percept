#include "deploy_percept/post_process/YoloV5DetectPostProcess.hpp"
#include "deploy_percept/post_process/types.hpp"
#include <vector>
#include <algorithm>
#include <set>
#include <cstring>

namespace deploy_percept
{
    namespace post_process
    {

        YoloV5DetectPostProcess::YoloV5DetectPostProcess(const YoloV5DetectPostProcess::Params &params)
            : YoloBasePostProcess(), params_(params)
        {
        }

        bool YoloV5DetectPostProcess::run(
            int8_t *input0,
            int8_t *input1,
            int8_t *input2,
            int model_in_h,
            int model_in_w,
            BoxRect pads,
            float scale_w,
            float scale_h,
            std::vector<int32_t> &qnt_zps,
            std::vector<float> &qnt_scales)
        {
            // 重置结果
            memset(&result_.group, 0, sizeof(DetectResultGroup));
            result_.success = false;
            result_.message = "";

            std::vector<float> filterBoxes;
            std::vector<float> objProbs;
            std::vector<int> classId;

            // stride 8
            int stride0 = 8;
            int grid_h0 = model_in_h / stride0;
            int grid_w0 = model_in_w / stride0;
            int validCount0 = 0;
            validCount0 = processYoloOutput(input0, params_.anchor_stride8.data(), grid_h0, grid_w0,
                                            model_in_h, model_in_w, stride0, filterBoxes, objProbs,
                                            classId, params_.conf_threshold, qnt_zps[0], qnt_scales[0]);

            // stride 16
            int stride1 = 16;
            int grid_h1 = model_in_h / stride1;
            int grid_w1 = model_in_w / stride1;
            int validCount1 = 0;
            validCount1 = processYoloOutput(input1, params_.anchor_stride16.data(), grid_h1, grid_w1,
                                            model_in_h, model_in_w, stride1, filterBoxes, objProbs,
                                            classId, params_.conf_threshold, qnt_zps[1], qnt_scales[1]);

            // stride 32
            int stride2 = 32;
            int grid_h2 = model_in_h / stride2;
            int grid_w2 = model_in_w / stride2;
            int validCount2 = 0;
            validCount2 = processYoloOutput(input2, params_.anchor_stride32.data(), grid_h2, grid_w2,
                                            model_in_h, model_in_w, stride2, filterBoxes, objProbs,
                                            classId, params_.conf_threshold, qnt_zps[2], qnt_scales[2]);

            int validCount = validCount0 + validCount1 + validCount2;
            // no object detect
            if (validCount <= 0)
            {
                result_.message = "No objects detected";
                return false;
            }

            std::vector<int> indexArray;
            for (int i = 0; i < validCount; ++i)
            {
                indexArray.push_back(i);
            }

            quickSortIndices(objProbs, 0, validCount - 1, indexArray);

            std::set<int> class_set(classId.begin(), classId.end());

            for (auto c : class_set)
            {
                nms(validCount, filterBoxes, classId, indexArray, c, params_.nms_threshold);
            }

            int last_count = 0;
            result_.group.count = 0;
            /* box valid detect target */
            for (int i = 0; i < validCount; ++i)
            {
                if (indexArray[i] == -1 || last_count >= params_.obj_numb_max_size)
                {
                    continue;
                }
                int n = indexArray[i];

                float x1 = filterBoxes[n * 4 + 0] - pads.left;
                float y1 = filterBoxes[n * 4 + 1] - pads.top;
                float x2 = x1 + filterBoxes[n * 4 + 2];
                float y2 = y1 + filterBoxes[n * 4 + 3];

                int id = classId[n];
                float obj_conf = objProbs[i];

                result_.group.results[last_count].box.left = static_cast<int>(clamp(x1, 0, model_in_w) / scale_w);
                result_.group.results[last_count].box.top = static_cast<int>(clamp(y1, 0, model_in_h) / scale_h);
                result_.group.results[last_count].box.right = static_cast<int>(clamp(x2, 0, model_in_w) / scale_w);
                result_.group.results[last_count].box.bottom = static_cast<int>(clamp(y2, 0, model_in_h) / scale_h);
                result_.group.results[last_count].prop = obj_conf;

                // 设置标签名称，这里只是框架，实际需要从外部加载标签
                snprintf(result_.group.results[last_count].name, params_.obj_name_max_size, "class_%d", id);

                last_count++;
            }
            result_.group.count = last_count;

            result_.success = (last_count > 0);
            result_.message = "Processing completed successfully";
            return true;
        }

        /**
         * @brief 在图像上绘制检测结果组
         * @param image 输入图像，会在该图像上直接绘制
         * @param detect_result_group 检测结果组，包含所有检测框信息
         * @param font_scale 字体缩放比例，默认为0.4
         * @param line_thickness 线条粗细，默认为3
         */
        void YoloV5DetectPostProcess::drawDetectionsResultGroupOnImage(cv::Mat& image, 
                                                                    const DetectResultGroup& detect_result_group,
                                                                    double font_scale,
                                                                    int line_thickness)
        {
            char text[256];
            for (int i = 0; i < detect_result_group.count; i++)
            {
                const auto &det_result = detect_result_group.results[i];
                sprintf(text, "%s %.1f%%", det_result.name, det_result.prop * 100);
                printf("%s @ (%d %d %d %d) %f\n", det_result.name, det_result.box.left, det_result.box.top,
                       det_result.box.right, det_result.box.bottom, det_result.prop);
                cv::rectangle(image, cv::Point(det_result.box.left, det_result.box.top), cv::Point(det_result.box.right, det_result.box.bottom), cv::Scalar(256, 0, 0, 256), line_thickness);
                cv::putText(image, text, cv::Point(det_result.box.left, det_result.box.top + 12), cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255));
            }
        }

        /**
         * @brief 处理YoloV5模型的单个输出层
         *
         * 解析模型输出的特征图，提取目标边界框、置信度和类别信息
         *
         * @param input 模型输出数据指针
         * @param anchor 当前尺度的锚框尺寸
         * @param grid_h 特征图网格高度
         * @param grid_w 特征图网格宽度
         * @param height 原始输入图像高度
         * @param width 原始输入图像宽度
         * @param stride 特征图相对于原图的缩放步长
         * @param boxes 输出的边界框坐标集合 (x, y, w, h)
         * @param objProbs 输出的目标置信度集合
         * @param classId 输出的类别ID集合
         * @param threshold 置信度阈值
         * @param zp 量化零点参数
         * @param scale 量化缩放参数
         * @return 有效检测框的数量
         */
        int YoloV5DetectPostProcess::processYoloOutput(int8_t *input, int *anchor, int grid_h, int grid_w,
                                                       int height, int width, int stride,
                                                       std::vector<float> &boxes, std::vector<float> &objProbs,
                                                       std::vector<int> &classId, float threshold,
                                                       int32_t zp, float scale)
        {
            int validCount = 0;                                     // 有效检测框计数
            int grid_len = grid_h * grid_w;                         // 当前特征图的网格总数
            int8_t thres_i8 = qntF32ToAffine(threshold, zp, scale); // 将浮点阈值转换为量化后的整数值
            const int prop_box_size = (5 + params_.obj_class_num);  // 每个网格预测的属性数(4个坐标+1个置信度+类别数)
            // 按锚框3，xywh执行度5，类别排序80
            // 遍历每个锚框（YOLO通常每个网格有3个不同尺寸的锚框）
            for (int a = 0; a < 3; a++)
            {
                // 遍历特征图上的每个网格点
                for (int i = 0; i < grid_h; i++)
                { // i: 网格的y坐标
                    for (int j = 0; j < grid_w; j++)
                    { // j: 网格的x坐标
                        // 获取当前网格位置的边界框置信度（第4个位置是置信度）
                        int index = (prop_box_size * a + 4) * grid_len + i * grid_w + j;
                        int8_t box_confidence = input[(prop_box_size * a + 4) * grid_len + i * grid_w + j];
                        // a,i,j-index_confidence
                        // 0,0,0-25600
                        // 0,0,1-25601
                        // 如果置信度超过阈值，则认为检测到目标
                        if (box_confidence >= thres_i8)
                        {
                            // 计算当前锚框在输出张量中的偏移量
                            int offset = (prop_box_size * a) * grid_len + i * grid_w + j;
                            int8_t *in_ptr = input + offset;
                            // 缩放和偏移：为了增加模型的灵活性和提升训练稳定性，YOLOv5 对输出进行了缩放和偏移处理。公式 * 2.0 - 0.5 的含义是：

                            // * 2.0：将范围从 [0, 1] 扩展到 [0, 2]
                            // - 0.5：将范围从 [0, 2] 移动到 [-0.5, 1.5]
                            // 这样，经过 * 2.0 - 0.5 变换后的值范围是 [-0.5, 1.5]，这使得模型能够预测超出单个网格单元的边界框中心点，从而提高对目标位置的预测灵活性。具体来说：

                            // 当输出为 0 时，经过变换后为 -0.5，表示中心点可以稍微超出网格单元的左边界
                            // 当输出为 1 时，经过变换后为 1.5，表示中心点可以稍微超出网格单元的右边界
                            // 解析边界框坐标（YOLOv5使用中心点坐标+宽高格式）
                            float box_x = (deqntAffineToF32(*in_ptr, zp, scale)) * 2.0 - 0.5;          // 中心点x坐标
                            float box_y = (deqntAffineToF32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5; // 中心点y坐标
                            float box_w = (deqntAffineToF32(in_ptr[2 * grid_len], zp, scale)) * 2.0;   // 宽度
                            float box_h = (deqntAffineToF32(in_ptr[3 * grid_len], zp, scale)) * 2.0;   // 高度

                            // 将归一化坐标转换为原图坐标
                            box_x = (box_x + j) * (float)stride; // 加上网格位置并缩放到原图尺寸
                            box_y = (box_y + i) * (float)stride;

                            // 使用锚框尺寸缩放边界框的宽高
                            box_w = box_w * box_w * (float)anchor[a * 2];     // 使用锚框宽度
                            box_h = box_h * box_h * (float)anchor[a * 2 + 1]; // 使用锚框高度

                            // 计算边界框左上角坐标
                            box_x -= (box_w / 2.0);
                            box_y -= (box_h / 2.0);

                            // 寻找最大类别概率和对应的类别ID
                            int8_t maxClassProbs = in_ptr[5 * grid_len]; // 第一个类别的概率
                            int maxClassId = 0;
                            for (int k = 1; k < params_.obj_class_num; ++k)
                            {                                             // 遍历所有类别
                                int8_t prob = in_ptr[(5 + k) * grid_len]; // 获取第k个类别的概率
                                if (prob > maxClassProbs)
                                {
                                    maxClassId = k;       // 更新最大概率的类别ID
                                    maxClassProbs = prob; // 更新最大概率值
                                }
                            }

                            // 如果最大类别概率超过阈值，则确认检测到目标
                            if (maxClassProbs > thres_i8)
                            {
                                // 计算最终置信度：边界框置信度 × 最大类别概率
                                objProbs.push_back((deqntAffineToF32(maxClassProbs, zp, scale)) *
                                                   (deqntAffineToF32(box_confidence, zp, scale)));
                                classId.push_back(maxClassId); // 记录类别ID
                                validCount++;                  // 有效检测计数+1

                                // 记录边界框坐标
                                boxes.push_back(box_x); // 左上角x坐标
                                boxes.push_back(box_y); // 左上角y坐标
                                boxes.push_back(box_w); // 宽度
                                boxes.push_back(box_h); // 高度
                            }
                        }
                    }
                }
            }
            return validCount; // 返回有效检测框数量
        }

        void YoloV5DetectPostProcess::quickSortIndices(std::vector<float> &input, int left, int right, std::vector<int> &indices)
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
                quickSortIndices(input, left, low - 1, indices);
                quickSortIndices(input, low + 1, right, indices);
            }
        }

    } // namespace post_process
} // namespace deploy_percept