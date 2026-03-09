#include <iostream>
#include <stdio.h>
#include <string>
#include <memory>
#include <vector>
#include <sys/time.h>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <fstream>

#include "rknn_api.h"

#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "cnpy.h"

#include "deploy_percept/post_process/YoloV8SegPostProcess.hpp"
#include "deploy_percept/post_process/types.hpp"
#include "deploy_percept/engine/RknnEngine.hpp"
#include "deploy_percept/utils/npy.hpp"
#include "deploy_percept/utils/io.hpp"

using namespace deploy_percept::utils;
double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

/**
 * @brief 将RKNN输出数组转换为NPZ格式
 * @param outputs RKNN输出数组指针
 * @param output_count 输出tensor数量
 * @param output_attrs 输出tensor属性向量
 * @param prefix 键名前缀，默认为"output_"
 * @return cnpy::npz_t 转换后的NPZ对象
 * @throws std::runtime_error 当转换过程中发生错误时抛出异常
 */
cnpy::npz_t convertRknnOutputsToNpz(const rknn_output *outputs,
                                    uint32_t output_count,
                                    const std::vector<rknn_tensor_attr> &output_attrs,
                                    const std::string &prefix = "output_")
{
    cnpy::npz_t npz_result;

    for (uint32_t i = 0; i < output_count; ++i)
    {
        const auto &attr = output_attrs[i];

        std::string key = prefix + std::to_string(i);

        size_t data_size = attr.n_elems;
        const int8_t *data_ptr = static_cast<const int8_t *>(outputs[i].buf);

        std::vector<size_t> shape(attr.dims, attr.dims + attr.n_dims);
        cnpy::NpyArray array(shape, sizeof(int8_t), false); // false表示C-order

        std::memcpy(array.data<int8_t>(), data_ptr, data_size * sizeof(int8_t));

        npz_result[key] = std::move(array);
    }

    return npz_result;
}

int main()
{
    std::string path_model_rknn = "runs/models/RK3588/yolov8_seg.rknn";
    std::string path_input_img = "apps/yolov8_seg_rknn/bus.jpg";
    std::filesystem::path path_model_output_npz = "apps/yolov8_seg_rknn/yolov8_seg_result_model_outputs.npz";

    std::string path_save_output_img = "tmp/yolov8_seg_out.jpg";
    std::filesystem::path path_save_model_output_npz = "tmp/yolov8_seg_result_model_outputs.npz";
    std::filesystem::path path_save_mask_bin = "tmp/yolov8_seg_mask.bin";

    deploy_percept::engine::RknnEngine::Params params;
    params.model_path = path_model_rknn;

    deploy_percept::engine::RknnEngine engine(params);

    cv::Mat orig_img = cv::imread(path_input_img, 1);

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

    cnpy::npz_t expected_model_outputs_npz = cnpy::npz_load(path_model_output_npz);
    cnpy::npz_t model_outputs_npz = convertRknnOutputsToNpz(
        outputs,
        engine.model_io_num_.n_output,
        engine.model_output_attrs_);
    bool validation_result = deploy_percept::utils::areNpzObjectsIdentical(expected_model_outputs_npz, model_outputs_npz);
    if (validation_result)
    {
        printf("Model output same.\n");
    }

    if (save_npz(path_save_model_output_npz, model_outputs_npz))
    {
        printf("save outputs to npz. %s\n", path_save_model_output_npz.c_str());
    }

    std::vector<int8_t *> output_buffers_int8;
    std::vector<std::vector<int>> output_dims;
    std::vector<float> output_scales;
    std::vector<int32_t> output_zps;

    for (int i = 0; i < engine.model_io_num_.n_output; i++)
    {
        output_buffers_int8.push_back(static_cast<int8_t *>(outputs[i].buf));

        std::vector<int> dims(4);
        dims[0] = engine.model_output_attrs_[i].dims[0];
        dims[1] = engine.model_output_attrs_[i].dims[1];
        dims[2] = engine.model_output_attrs_[i].dims[2];
        dims[3] = engine.model_output_attrs_[i].dims[3];
        output_dims.push_back(dims);
        output_scales.push_back(engine.model_output_attrs_[i].scale);
        output_zps.push_back(engine.model_output_attrs_[i].zp);
    }

    deploy_percept::post_process::YoloV8SegPostProcess::Params params_post;
    deploy_percept::post_process::YoloV8SegPostProcess seg_processor(params_post);
    // output_dims 这个参数可以省略
    bool success = seg_processor.run(
        output_buffers_int8,
        orig_img.cols,
        orig_img.rows,
        output_dims,
        output_scales,
        output_zps);

    rknn_outputs_release(engine.ctx_, engine.model_io_num_.n_output, outputs);

    auto seg_results = seg_processor.getResult().group;

    cv::Mat result_img = orig_img.clone();
    seg_processor.drawDetectionResults(result_img, seg_results);

    cv::imwrite(path_save_output_img, result_img);
    printf("Save computed detect result to %s\n", path_save_output_img.c_str());

    if (saveUint8VectorToBinFile(seg_results.segmentation_mask, path_save_mask_bin))
    {
        printf("Save computed segmentation mask to %s\n", path_save_mask_bin.c_str());
    }

    return 0;
}