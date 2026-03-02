#include "deploy_percept/post_process/YoloV8SegPostProcess.hpp"
#include "deploy_percept/utils/npy.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>
#include "cnpy.h"

using namespace deploy_percept::post_process;
namespace fs = std::filesystem;

class YoloV8SegPostProcessTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        YoloV8SegPostProcess::Params params;
        processor = std::make_unique<YoloV8SegPostProcess>(params);
        fs::path path_model_outputs_npz = "/home/orangepi/HectorHuang/deploy_percept/examples/data/yolov8_seg/yolov8_seg_outputs.npz";
        model_outputs_npz = cnpy::npz_load(path_model_outputs_npz);
        path_seg_result = "/home/orangepi/HectorHuang/deploy_percept/examples/data/yolov8_seg/yolov8_seg_mask_0.bin";
    }

    void TearDown() override
    {
    }

    std::unique_ptr<deploy_percept::post_process::YoloV8SegPostProcess> processor;
    cnpy::npz_t model_outputs_npz;
    fs::path path_seg_result;
};

TEST_F(YoloV8SegPostProcessTest, run)

{
    // 读取expected_seg
    std::ifstream file(path_seg_result, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    SegmentationResult expected_seg;
    expected_seg.seg_mask.resize(size);
    file.read(reinterpret_cast<char *>(expected_seg.seg_mask.data()), size);
    file.close();

    auto make_detect = [](int cls_id, const char *name_str, float conf,
                          int x1, int y1, int x2, int y2) -> DetectResult
    {
        DetectResult res;
        res.cls_id = cls_id;
        res.prop = conf;
        res.box = {x1, y1, x2, y2};
        // 安全拷贝字符串到固定长度数组
        strncpy(res.name, name_str, sizeof(res.name) - 1);
        res.name[sizeof(res.name) - 1] = '\0';
        return res;
    };
    std::vector<DetectResult> expected_results = {
        make_detect(5, "class_5", 0.9113f, 87, 137, 553, 439),
        make_detect(0, "class_0", 0.8998f, 108, 236, 227, 537),
        make_detect(0, "class_0", 0.8693f, 211, 241, 283, 508),
        make_detect(0, "class_0", 0.8655f, 477, 232, 559, 519),
        make_detect(0, "class_0", 0.5403f, 79, 327, 125, 514),
        make_detect(27, "class_27", 0.2741f, 248, 284, 259, 310)};

    std::vector<std::vector<int>> output_dims;
    std::vector<float> output_scales;
    std::vector<int32_t> output_zps;
    std::vector<void *> output_buffers;
    output_dims = {
        {1, 64, 80, 80},
        {1, 80, 80, 80},
        {1, 1, 80, 80},
        {1, 32, 80, 80},
        {1, 64, 40, 40},
        {1, 80, 40, 40},
        {1, 1, 40, 40},
        {1, 32, 40, 40},
        {1, 64, 20, 20},
        {1, 80, 20, 20},
        {1, 1, 20, 20},
        {1, 32, 20, 20},
        {1, 32, 160, 160}};

    // 填充scale数据
    output_scales = {
        0.115401f, 0.003514f, 0.003540f, 0.019863f,
        0.099555f, 0.003555f, 0.003680f, 0.022367f,
        0.074253f, 0.003813f, 0.003922f, 0.019919f,
        0.032336f};

    // 填充zero point数据
    output_zps = {
        -61, -128, -128, 27,
        -15, -128, -128, 30,
        -55, -128, -128, 43,
        -119};

    output_buffers.reserve(13);
    for (int i = 0; i < 13; ++i)
    {
        std::string key = "output_" + std::to_string(i);
        auto it = model_outputs_npz.find(key);
        ASSERT_NE(it, model_outputs_npz.end()) << "Key not found: " << key;

        // 获取数组数据指针（void* 指向原始数据）
        void *data_ptr = it->second.data<void>();
        output_buffers.push_back(data_ptr);
    }
    // --- 调用被测方法 ---
    bool success = processor->run(&output_buffers, 640, 640,
                                  output_dims, output_scales, output_zps);
    ASSERT_TRUE(success);
    const auto &result = processor->getResult();
    ASSERT_TRUE(result.success);

    const auto &result_group = result.group;

    // --- 比较检测结果 ---
    for (size_t i = 0; i < expected_results.size(); ++i)
    {
        const auto &exp = expected_results[i];
        const auto &act = result_group.results[i];
        EXPECT_EQ(exp.cls_id, act.cls_id);
        EXPECT_STREQ(exp.name, act.name);
        EXPECT_NEAR(exp.prop, act.prop, 1e-4);
        EXPECT_EQ(exp.box.left, act.box.left);
        EXPECT_EQ(exp.box.top, act.box.top);
        EXPECT_EQ(exp.box.right, act.box.right);
        EXPECT_EQ(exp.box.bottom, act.box.bottom);
    }

    // --- 比较分割结果（假设只有一个分割输出）---
    ASSERT_EQ(result_group.results_seg.size(), 1);
    const auto &actual_seg = result_group.results_seg[0].seg_mask;
    ASSERT_EQ(actual_seg.size(), expected_seg.seg_mask.size());
    for (size_t i = 0; i < actual_seg.size(); ++i)
    {
        EXPECT_EQ(actual_seg[i], expected_seg.seg_mask[i]) << "Mismatch at index " << i;
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}