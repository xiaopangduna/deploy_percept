#pragma once

#include <vector>
#include <cstring>
#include <string>
#include <opencv2/opencv.hpp>
#include "deploy_percept/post_process/YoloBasePostProcess.hpp"
#include "deploy_percept/post_process/types.hpp"

namespace deploy_percept
{
    namespace post_process
    {
        class YoloV5SegPostProcess : public YoloBasePostProcess
        {
        public:
            // 参数配置结构体
            struct Params
            {
                float conf_threshold = 0.25f;
                float nms_threshold = 0.45f;
                int obj_class_num = 80;
                int obj_name_max_size = 16;
                int obj_numb_max_size = 64;

                // 分割相关参数
                int proto_channel = 32;
                int proto_height = 160;
                int proto_weight = 160;
                
                // Anchor参数
                std::vector<int> anchor_stride8 = {10, 13, 16, 30, 33, 23};
                std::vector<int> anchor_stride16 = {30, 61, 62, 45, 59, 119};
                std::vector<int> anchor_stride32 = {116, 90, 156, 198, 373, 326};
                
                // 模型结构参数
                int prop_box_size = 5;  // 边界框坐标(xywh) + 置信度
            };

            // 分割结果结构体
            struct SegmentationResult {
                uint8_t *seg_mask = nullptr;
            };

            struct SegmentationResultGroup {
                int id = 0;
                int count = 0;
                std::vector<DetectResult> results; // 检测结果
                std::vector<SegmentationResult> results_seg; // 分割结果
            };

            // 结果结构体
            struct Result
            {
                SegmentationResultGroup group{};
                bool success = false;
                std::string message{}; // 可选的详细信息
            };

            // 使用参数结构体的构造函数
            explicit YoloV5SegPostProcess(const Params &params);
            ~YoloV5SegPostProcess() = default;
            const Params &getParams() const { return params_; }
            const Result &getResult() const { return result_; }

            bool run(
                std::vector<std::vector<int>> &output_dims,
                std::vector<float> &output_scales,
                std::vector<int32_t> &output_zps,
                std::vector<void*> *outputs,
                int input_image_width,
                int input_image_height);

        private:
            Params params_;
            Result result_{};
            
            int process_i8(std::vector<void*> *all_input, int input_id, int *anchor, int grid_h, int grid_w, 
                    int stride,
                    std::vector<float> &boxes, std::vector<float> &segments,
                    std::vector<float> &objProbs, std::vector<int> &classId, float threshold,
                    std::vector<std::vector<int>> &output_dims, std::vector<float> &output_scales, 
                    std::vector<int32_t> &output_zps);
            static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices);
            
                                      
                           
            static int clamp(float val, int min, int max);
            
            static void matmul_by_cpu_uint8(std::vector<float> &A, float *B, uint8_t *C, int ROWS_A, int COLS_A, int COLS_B);
            
            static void resize_by_opencv_uint8(uint8_t *input_image, int input_width, int input_height, int boxes_num, 
                                             uint8_t *output_image, int target_width, int target_height);
                                             
            static void crop_mask_uint8(uint8_t *seg_mask, uint8_t *all_mask_in_one, float *boxes, int boxes_num, 
                                      int *cls_id, int height, int width);
                                      
            static void seg_reverse(uint8_t *seg_mask, uint8_t *cropped_seg, uint8_t *seg_mask_real,
                                  int input_image_height, int input_image_width, int cropped_height, int cropped_width, 
                                  int ori_in_height, int ori_in_width, int y_pad, int x_pad);
        };
    } // namespace post_process
} // namespace deploy_percept