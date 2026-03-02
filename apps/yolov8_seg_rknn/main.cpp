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

    // 打印后处理结果用于调试
    printf("\n=== 后处理结果 ===\n");
    printf("检测到的目标数量: %d\n", seg_results.count);
    printf("图像ID: %d\n", seg_results.id);
    
    for (int i = 0; i < seg_results.count; ++i) {
        const auto& detect_result = seg_results.results[i];
        const auto& box = detect_result.box;
        
        printf("目标 %d:\n", i + 1);
        printf("  类别ID: %d\n", detect_result.cls_id);
        printf("  类别名称: %s\n", detect_result.name);
        printf("  置信度: %.4f\n", detect_result.prop);
        printf("  边界框: [%d, %d, %d, %d]\n", box.left, box.top, box.right, box.bottom);
        printf("  宽度: %d, 高度: %d\n", box.right - box.left, box.bottom - box.top);
        
        // 如果有分割掩码，打印相关信息
        if (i < static_cast<int>(seg_results.results_seg.size()) && 
            !seg_results.results_seg[i].seg_mask.empty()) {
            printf("  分割掩码大小: %zu 字节\n", seg_results.results_seg[i].seg_mask.size());
        }
        printf("\n");
    }
    printf("==================\n\n");

    // 绘制计算得到的检测结果 - 使用类的成员函数
    cv::Mat result_img = orig_img.clone();
    seg_processor.drawDetectionResults(result_img, seg_results);

    std::string computed_out_path = "/home/orangepi/HectorHuang/deploy_percept/tmp/yolov8_seg_out.jpg";
    printf("Save computed detect result to %s\n", computed_out_path.c_str());
    cv::imwrite(computed_out_path, result_img);
    rknn_outputs_release(engine.ctx_, engine.model_io_num_.n_output, outputs);

    return 0;
}