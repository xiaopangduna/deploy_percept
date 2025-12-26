#ifndef DEPLOY_PERCEPT_POST_PROCESS_YOLOV5DETECTPOSTPROCESS_HPP
#define DEPLOY_PERCEPT_POST_PROCESS_YOLOV5DETECTPOSTPROCESS_HPP

#include <vector>
#include "deploy_percept/post_process/YoloBasePostProcess.hpp"
#include "deploy_percept/post_process/types.hpp"

namespace deploy_percept
{
    namespace post_process
    {

        class YoloV5DetectPostProcess : public YoloBasePostProcess
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

                std::vector<int> anchor_stride8 = {10, 13, 16, 30, 33, 23};
                std::vector<int> anchor_stride16 = {30, 61, 62, 45, 59, 119};
                std::vector<int> anchor_stride32 = {116, 90, 156, 198, 373, 326};
            };

            // 使用参数结构体的构造函数
            explicit YoloV5DetectPostProcess(const Params &params );
            ~YoloV5DetectPostProcess() = default;

            // YOLOv5特定的处理接口
            int process(
                int8_t *input0,
                int8_t *input1,
                int8_t *input2,
                int model_in_h,
                int model_in_w,
                BoxRect pads,
                float scale_w,
                float scale_h,
                std::vector<int32_t> &qnt_zps,
                std::vector<float> &qnt_scales,
                DetectResultGroup *group);

            int processYoloOutput(int8_t *input, int *anchor, int grid_h, int grid_w,
                                  int height, int width, int stride,
                                  std::vector<float> &boxes, std::vector<float> &objProbs,
                                  std::vector<int> &classId, float threshold,
                                  int32_t zp, float scale);

            // 获取当前参数的方法
            const Params &getParams() const { return params_; }

        private:
            // YoloV5特有的一些处理函数
            void quickSortIndices(std::vector<float> &input, int left, int right, std::vector<int> &indices);

            // 参数配置
            Params params_;
        };

    } // namespace post_process
} // namespace deploy_percept

#endif // DEPLOY_PERCEPT_POST_PROCESS_YOLOV5DETECTPOSTPROCESS_HPP