#include <cstdio>
#include <cstring>
#include <filesystem>
#include <sys/time.h>
#include <vector>

#include "rknn_api.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "deploy_percept/engine/RknnEngine.hpp"
#include "deploy_percept/post_process/YoloV8PosePostProcess.hpp"
#include "deploy_percept/post_process/types.hpp"

using namespace deploy_percept::post_process;

static double __get_us(struct timeval t)
{
    return static_cast<double>(t.tv_sec * 1000000 + t.tv_usec);
}

int main()
{
    const std::string path_model_rknn = "runs/models/RK3588/yolov8n_pose.rknn";
    const std::string path_input_img = "apps/yolov8_pose_rknn/bus.jpg";
    const std::string path_save_output_img = "tmp/yolov8_pose_out.jpg";

    deploy_percept::engine::RknnEngine::Params params;
    params.model_path = path_model_rknn;

    deploy_percept::engine::RknnEngine engine(params);

    cv::Mat orig_img = cv::imread(path_input_img, 1);
    if (orig_img.empty())
    {
        fprintf(stderr, "Failed to read image: %s\n", path_input_img.c_str());
        return 1;
    }

    cv::Mat img;
    cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);

    const int in_w = static_cast<int>(engine.model_input_attrs_[0].dims[2]);
    const int in_h = static_cast<int>(engine.model_input_attrs_[0].dims[1]);

    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(in_w, in_h));

    rknn_input inputs[engine.model_io_num_.n_input];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size =
        static_cast<uint32_t>(engine.model_input_attrs_[0].dims[1] * engine.model_input_attrs_[0].dims[2] *
                              engine.model_input_attrs_[0].dims[3]);
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    inputs[0].buf = resized_img.data;

    rknn_output outputs[engine.model_io_num_.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (uint32_t i = 0; i < engine.model_io_num_.n_output; i++)
    {
        outputs[i].index = i;
        outputs[i].want_float = 0;
    }

    struct timeval start_time, stop_time;
    gettimeofday(&start_time, nullptr);
    engine.run(inputs, outputs);
    gettimeofday(&stop_time, nullptr);
    printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000.0);

    std::vector<void *> output_ptrs;
    std::vector<std::vector<int>> output_dims;
    std::vector<float> output_scales;
    std::vector<int32_t> output_zps;
    std::vector<int32_t> output_types;

    for (uint32_t i = 0; i < engine.model_io_num_.n_output; i++)
    {
        output_ptrs.push_back(outputs[i].buf);

        std::vector<int> dims(engine.model_output_attrs_[i].n_dims);
        for (uint32_t j = 0; j < engine.model_output_attrs_[i].n_dims; ++j)
        {
            dims[j] = static_cast<int>(engine.model_output_attrs_[i].dims[j]);
        }
        output_dims.push_back(std::move(dims));
        output_scales.push_back(engine.model_output_attrs_[i].scale);
        output_zps.push_back(engine.model_output_attrs_[i].zp);
        output_types.push_back(static_cast<int32_t>(engine.model_output_attrs_[i].type));
    }

    YoloV8PosePostProcess::Params ppose;
    YoloV8PosePostProcess pose_processor(ppose);

    const bool ok = pose_processor.run(output_ptrs,
                                       in_w,
                                       in_h,
                                       output_dims,
                                       output_scales,
                                       output_zps,
                                       output_types);

    rknn_outputs_release(engine.ctx_, engine.model_io_num_.n_output, outputs);

    if (!ok)
    {
        fprintf(stderr, "YoloV8PosePostProcess::run failed\n");
        return 1;
    }

    const PoseResultGroup &pose_results = pose_processor.getResult().group;
    printf("pose_results.count=%d\n", pose_results.count);
    for (int i = 0; i < pose_results.count && i < 5; ++i)
    {
        printf("  [%d] cls=%d prop=%.3f box=(%d,%d)-(%d,%d)\n",
               i,
               pose_results.objects[i].cls_id,
               pose_results.objects[i].prop,
               pose_results.objects[i].box.left,
               pose_results.objects[i].box.top,
               pose_results.objects[i].box.right,
               pose_results.objects[i].box.bottom);
    }

    cv::Mat result_img = orig_img.clone();
    pose_processor.drawPoseResults(result_img, pose_results);

    std::filesystem::create_directories(std::filesystem::path(path_save_output_img).parent_path());
    cv::imwrite(path_save_output_img, result_img);
    printf("Save pose result to %s\n", path_save_output_img.c_str());

    return 0;
}
