#include <iostream>
#include <stdio.h>
#include <string>
#include <memory>
#include <vector>
#include <sys/time.h>
#include <chrono>
#include <filesystem>
#include <iomanip>

#include "rknn_api.h"

#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "cnpy.h"

#include "deploy_percept/post_process/YoloV8SegPostProcess.hpp"
#include "deploy_percept/post_process/types.hpp"
#include "deploy_percept/engine/RknnEngine.hpp"
#include "deploy_percept/utils/npy.hpp" // 包含新的工具函数

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

/**
 * @brief 验证模型输出与参考文件是否一致
 * @param outputs RKNN输出数组
 * @param engine RKNN引擎实例
 * @param reference_file 参考NPZ文件路径
 * @return bool true表示一致，false表示不一致
 */
bool validateModelOutput(const rknn_output *outputs,
                         const deploy_percept::engine::RknnEngine &engine,
                         const std::string &reference_file)
{
    try
    {
        if (!std::filesystem::exists(reference_file))
        {
            std::cerr << "Reference file not found: " << reference_file << std::endl;
            return false;
        }

        // 将当前RKNN输出转换为NPZ格式
        cnpy::npz_t current_npz;
        uint32_t output_count = engine.model_io_num_.n_output;

        for (uint32_t i = 0; i < output_count; ++i)
        {
            auto &attr = engine.model_output_attrs_[i];

            // 创建键名
            std::string key = "output_" + std::to_string(i);

            // 获取数据信息
            size_t data_size = attr.n_elems;
            const int8_t *data_ptr = static_cast<const int8_t *>(outputs[i].buf);

            // 创建NpyArray对象 - 使用正确的构造函数
            std::vector<size_t> shape(attr.dims, attr.dims + attr.n_dims);
            cnpy::NpyArray array(shape, sizeof(int8_t), false); // false表示C-order

            // 复制数据
            std::memcpy(array.data<int8_t>(), data_ptr, data_size * sizeof(int8_t));

            // 添加到NPZ对象
            current_npz[key] = std::move(array);
        }

        // 加载参考数据
        cnpy::npz_t reference_npz = cnpy::npz_load(reference_file);

        // 使用工具函数比较
        return deploy_percept::utils::areNpzObjectsIdentical(current_npz, reference_npz);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error validating model output: " << e.what() << std::endl;
        return false;
    }
}

int main()
{
    std::string model_name = "/home/orangepi/HectorHuang/deploy_percept/runs/models/RK3588/yolov8_seg.rknn";

    deploy_percept::engine::RknnEngine::Params params;
    params.model_path = model_name;

    deploy_percept::engine::RknnEngine engine(params);

    // 读取图片
    std::string input_path = "/home/orangepi/HectorHuang/deploy_percept/apps/yolov8_seg_rknn/bus.jpg";
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

    // 使用极简的验证接口
    std::cout << "\n=== Simple NPZ Validation ===" << std::endl;

    std::filesystem::path project_root = std::filesystem::current_path().parent_path().parent_path();
    std::filesystem::path reference_path = "/home/orangepi/HectorHuang/deploy_percept/examples/data/yolov8_seg/yolov8_seg_outputs.npz";

    // bool validation_result = validateModelOutput(outputs, engine, reference_path.string());

    // 为兼容新的post_process函数接口，创建std::vector<void*>并提取输出张量属性
    std::vector<void *> output_buffers;
    std::vector<std::vector<int>> output_dims;
    std::vector<float> output_scales;
    std::vector<int32_t> output_zps;

    for (int i = 0; i < engine.model_io_num_.n_output; i++)
    {
        // 添加输出缓冲区指针
        output_buffers.push_back(outputs[i].buf);

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
    deploy_percept::post_process::YoloV8SegPostProcess::Params params_post;
    deploy_percept::post_process::YoloV8SegPostProcess seg_processor(params_post);

    // 打印后处理参数用于调试
    printf("\n=== YoloV8SegPostProcess Parameters ===\n");
    printf("Image dimensions: %d x %d\n", orig_img.cols, orig_img.rows);
    printf("Number of output tensors: %zu\n", output_buffers.size());
    
    printf("Output dimensions:\n");
    for (size_t i = 0; i < output_dims.size(); i++) {
        printf("  Tensor[%zu]: [%d, %d, %d, %d]\n", i, 
               output_dims[i][0], output_dims[i][1], output_dims[i][2], output_dims[i][3]);
    }
    
    printf("Output scales:\n");
    for (size_t i = 0; i < output_scales.size(); i++) {
        printf("  Tensor[%zu]: %.6f\n", i, output_scales[i]);
    }
    
    printf("Output zero points:\n");
    for (size_t i = 0; i < output_zps.size(); i++) {
        printf("  Tensor[%zu]: %d\n", i, output_zps[i]);
    }
    
    printf("Output buffer pointers:\n");
    for (size_t i = 0; i < output_buffers.size(); i++) {
        printf("  Buffer[%zu]: %p\n", i, output_buffers[i]);
    }
    printf("=====================================\n\n");

    // // 调用新的后处理类
    bool success = seg_processor.run(
        &output_buffers,
        orig_img.cols,
        orig_img.rows,
        output_dims,
        output_scales,
        output_zps);

    if (!success)
    {
        printf("Post-processing failed: %s\n", seg_processor.getResult().message.c_str());
        return -1;
    }
    // 获取后处理结果
    auto seg_results = seg_processor.getResult().group;

    // 打印检测结果
    printf("\n=== Detection Results ===\n");
    printf("Total detections: %d\n", seg_results.count);
    for (int i = 0; i < seg_results.count; i++) {
        const auto& det = seg_results.results[i];
        printf("Detection %d:\n", i);
        printf("  Class ID: %d\n", det.cls_id);
        printf("  Class Name: %s\n", det.name);
        printf("  Confidence: %.4f\n", det.prop);
        printf("  Bounding Box: [%d, %d, %d, %d]\n", 
               det.box.left, det.box.top, det.box.right, det.box.bottom);
        printf("  Width: %d, Height: %d\n", 
               det.box.right - det.box.left, det.box.bottom - det.box.top);
    }
    printf("========================\n\n");

    // 保存分割结果到文件
    printf("=== Saving Segmentation Results ===\n");
    for (size_t i = 0; i < seg_results.results_seg.size(); i++) {
        const auto& seg = seg_results.results_seg[i];
        if (!seg.seg_mask.empty()) {
            std::string seg_filename = "/home/orangepi/HectorHuang/deploy_percept/tmp/segmentation_mask_" + std::to_string(i) + ".bin";
            FILE* seg_file = fopen(seg_filename.c_str(), "wb");
            if (seg_file) {
                size_t written = fwrite(seg.seg_mask.data(), sizeof(uint8_t), seg.seg_mask.size(), seg_file);
                fclose(seg_file);
                printf("Saved segmentation mask %zu to %s (size: %zu bytes)\n", 
                       i, seg_filename.c_str(), seg.seg_mask.size());
                
                // 同时保存一些元数据信息
                std::string meta_filename = "/home/orangepi/HectorHuang/deploy_percept/tmp/segmentation_meta_" + std::to_string(i) + ".txt";
                FILE* meta_file = fopen(meta_filename.c_str(), "w");
                if (meta_file) {
                    fprintf(meta_file, "Segmentation Mask %zu Metadata\n", i);
                    fprintf(meta_file, "Mask Size: %zu bytes\n", seg.seg_mask.size());
                    fprintf(meta_file, "Detection Index: %zu\n", i);
                    if (i < static_cast<size_t>(seg_results.count)) {
                        const auto& det = seg_results.results[i];
                        fprintf(meta_file, "Associated Class ID: %d\n", det.cls_id);
                        fprintf(meta_file, "Associated Class Name: %s\n", det.name);
                        fprintf(meta_file, "Associated Confidence: %.4f\n", det.prop);
                        fprintf(meta_file, "Associated BBox: [%d, %d, %d, %d]\n", 
                                det.box.left, det.box.top, det.box.right, det.box.bottom);
                    }
                    fclose(meta_file);
                    printf("Saved metadata to %s\n", meta_filename.c_str());
                }
            } else {
                printf("Failed to create segmentation file: %s\n", seg_filename.c_str());
            }
        }
    }
    printf("===============================\n\n");

    cv::Mat result_img = orig_img.clone();
    seg_processor.drawDetectionResults(result_img, seg_results);

    std::string computed_out_path = "/home/orangepi/HectorHuang/deploy_percept/tmp/yolov8_seg_out.jpg";
    printf("Save computed detect result to %s\n", computed_out_path.c_str());
    cv::imwrite(computed_out_path, result_img);
    rknn_outputs_release(engine.ctx_, engine.model_io_num_.n_output, outputs);

    return 0;
}