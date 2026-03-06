#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <memory>
#include <vector>
#include <sys/time.h>
#include <chrono>
#include <fstream>

#include "rknn_api.h"

#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "cnpy.h"

#include "deploy_percept/post_process/YoloV5SegPostProcess.hpp"
#include "deploy_percept/post_process/types.hpp"
#include "deploy_percept/engine/RknnEngine.hpp"
#include "deploy_percept/utils/npy.hpp" 
#include "deploy_percept/utils/io.hpp"
using namespace deploy_percept::utils;
double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

int main()
{
    std::string model_name = "/home/orangepi/HectorHuang/deploy_percept/runs/models/RK3588/yolov5s_seg.rknn";

    deploy_percept::engine::RknnEngine::Params params;
    params.model_path = model_name;

    deploy_percept::engine::RknnEngine engine(params);

    // 读取图片
    std::string input_path = "/home/orangepi/HectorHuang/deploy_percept/apps/yolov5_seg_rknn/bus.jpg";
    printf("Read %s ...\n", input_path.c_str());

    cv::Mat orig_img = cv::imread(input_path, 1);

    cv::Mat img;
    cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);

    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(engine.model_input_attrs_[0].dims[2], engine.model_input_attrs_[0].dims[1]));

    rknn_input inputs[engine.model_io_num_.n_input];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = engine.model_input_attrs_[0].dims[1] * engine.model_input_attrs_[0].dims[2] * engine.model_input_attrs_[0].dims[3];
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    inputs[0].buf = resized_img.data;

    rknn_output outputs[engine.model_io_num_.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < engine.model_io_num_.n_output; i++)
    {
        outputs[i].index = i;
        outputs[i].want_float = 0;
    }

    struct timeval start_time, stop_time;
    gettimeofday(&start_time, NULL);
    engine.run(inputs, outputs);
    gettimeofday(&stop_time, NULL);
    printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

    // 保存outputs为NPZ格式
    {
        std::string output_path = "/home/orangepi/HectorHuang/deploy_percept/tmp/yolov5_seg_outputs.npz";

        for (int i = 0; i < engine.model_io_num_.n_output; i++)
        {
            std::string key = "output_" + std::to_string(i);
            std::vector<size_t> shape = {
                static_cast<size_t>(engine.model_output_attrs_[i].dims[0]),
                static_cast<size_t>(engine.model_output_attrs_[i].dims[1]),
                static_cast<size_t>(engine.model_output_attrs_[i].dims[2]),
                static_cast<size_t>(engine.model_output_attrs_[i].dims[3])};

            // 逐个保存每个输出张量
            std::string mode = (i == 0) ? "w" : "a"; // 第一个用"w"覆盖，后续用"a"追加
            cnpy::npz_save(output_path, key, static_cast<int8_t *>(outputs[i].buf), shape, mode);
        }

        printf("Saved outputs to %s\n", output_path.c_str());
    }

    std::vector<int8_t *> output_buffers_int8;
    std::vector<std::vector<int>> output_dims;
    std::vector<float> output_scales;
    std::vector<int32_t> output_zps;

    for (int i = 0; i < engine.model_io_num_.n_output; i++)
    {
        output_buffers_int8.push_back(static_cast<int8_t *>(outputs[i].buf));

        // 提取张量维度信息
        std::vector<int> dims(4);
        dims[0] = engine.model_output_attrs_[i].dims[0];
        dims[1] = engine.model_output_attrs_[i].dims[1];
        dims[2] = engine.model_output_attrs_[i].dims[2];
        dims[3] = engine.model_output_attrs_[i].dims[3];
        output_dims.push_back(dims);
        output_scales.push_back(engine.model_output_attrs_[i].scale);
        output_zps.push_back(engine.model_output_attrs_[i].zp);
    }

    // 使用YoloV5SegPostProcess类进行后处理
    deploy_percept::post_process::YoloV5SegPostProcess::Params params_post;
    deploy_percept::post_process::YoloV5SegPostProcess seg_processor(params_post);

    // // 调用新的后处理类
    bool success = seg_processor.run(
        output_buffers_int8,
        orig_img.cols,
        orig_img.rows,
        output_dims,
        output_scales,
        output_zps);

    // 获取后处理结果
    auto seg_results = seg_processor.getResult().group;

    std::filesystem::path path_save_mask_bin = "tmp/yolov5_seg_mask.bin";
    saveUint8VectorToBinFile(seg_results.segmentation_mask, path_save_mask_bin);

    // 绘制计算得到的检测结果 - 使用类的成员函数
    cv::Mat result_img = orig_img.clone();
    seg_processor.drawDetectionResults(result_img, seg_results);

    std::string computed_out_path = "apps/yolov5_seg_rknn/yolov5_seg_result.jpg";
    printf("Save computed detect result to %s\n", computed_out_path.c_str());
    cv::imwrite(computed_out_path, result_img);

    rknn_outputs_release(engine.ctx_, engine.model_io_num_.n_output, outputs);

    return 0;
}