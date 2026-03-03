#include "deploy_percept/post_process/YoloV8SegPostProcess.hpp"
#include "deploy_percept/utils/npy.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>
#include "cnpy.h"

using namespace deploy_percept::post_process;
using namespace deploy_percept::utils;
namespace fs = std::filesystem;

bool CompareDetectResultVectors(const std::vector<DetectionObject> &expected,
                                const std::vector<DetectionObject> &actual)
{
    bool match = true;
    // if (expected.size() != actual.size()) {
    //     EXPECT_EQ(expected.size(), actual.size());
    //     return false;
    // }
    for (size_t i = 0; i < expected.size(); ++i)
    {
        const auto &exp = expected[i];
        const auto &act = actual[i];
        // 使用 SCOPED_TRACE 帮助定位哪个元素失败
        ::testing::ScopedTrace trace(__FILE__, __LINE__,
                                     "Comparing element " + std::to_string(i));
        if (exp.cls_id != act.cls_id)
        {
            EXPECT_EQ(exp.cls_id, act.cls_id);
            match = false;
        }
        if (std::strcmp(exp.name, act.name) != 0)
        {
            EXPECT_STREQ(exp.name, act.name);
            match = false;
        }
        if (std::abs(exp.prop - act.prop) > 1e-4)
        {
            EXPECT_NEAR(exp.prop, act.prop, 1e-4);
            match = false;
        }
        if (exp.box.left != act.box.left ||
            exp.box.top != act.box.top ||
            exp.box.right != act.box.right ||
            exp.box.bottom != act.box.bottom)
        {
            EXPECT_EQ(exp.box.left, act.box.left);
            EXPECT_EQ(exp.box.top, act.box.top);
            EXPECT_EQ(exp.box.right, act.box.right);
            EXPECT_EQ(exp.box.bottom, act.box.bottom);
            match = false;
        }
    }
    return match;
}

// 比较两个分割掩码向量是否相等
bool CompareSegmentationMaskVectors(const std::vector<std::vector<uint8_t>>& expected,
                                   const std::vector<std::vector<uint8_t>>& actual) {
    bool match = true;
    if (expected.size() != actual.size()) {
        EXPECT_EQ(expected.size(), actual.size());
        return false;
    }
    
    for (size_t i = 0; i < expected.size(); ++i) {
        const auto& exp_mask = expected[i];
        const auto& act_mask = actual[i];
        
        ::testing::ScopedTrace trace(__FILE__, __LINE__, 
            "Comparing segmentation mask " + std::to_string(i));
            
        EXPECT_EQ(exp_mask.size(), act_mask.size());
        if (exp_mask.size() != act_mask.size()) {
            match = false;
            continue;
        }
        
        // 逐字节比较
        for (size_t j = 0; j < exp_mask.size(); ++j) {
            SCOPED_TRACE("Mask " + std::to_string(i) + " byte index " + std::to_string(j));
            EXPECT_EQ(exp_mask[j], act_mask[j]);
            if (::testing::Test::HasFailure()) {
                match = false;
            }
        }
    }
    return match;
}

std::vector<uint8_t> LoadSegmentationResult(const fs::path &file_path)
{
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        throw std::runtime_error("Cannot open file: " + file_path.string());
    }

    std::streamsize size = file.tellg();
    if (size < 0)
    {
        throw std::runtime_error("Failed to get file size: " + file_path.string());
    }
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> seg_mask;
    seg_mask.resize(static_cast<size_t>(size));

    if (!file.read(reinterpret_cast<char *>(seg_mask.data()), size))
    {
        throw std::runtime_error("Failed to read file: " + file_path.string());
    }

    // 可选：检查是否完整读取
    if (file.gcount() != size)
    {
        throw std::runtime_error("Read incomplete: " + file_path.string());
    }

    return seg_mask; // 返回vector<uint8_t>而不是SegmentationResult
}

class YoloV8SegPostProcessTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        YoloV8SegPostProcess::Params params;
        processor = std::make_unique<YoloV8SegPostProcess>(params);
        fs::path path_model_outputs_npz = "apps/yolov8_seg_rknn/yolov8_seg_result_model_outputs.npz";
        model_outputs_npz = cnpy::npz_load(path_model_outputs_npz);
        path_seg_result = "apps/yolov8_seg_rknn/yolov8_seg_result_mask.bin";
    }

    void TearDown() override
    {
    }

    std::unique_ptr<deploy_percept::post_process::YoloV8SegPostProcess> processor;
    cnpy::npz_t model_outputs_npz;
    fs::path path_seg_result;
    static DetectionObject MakeDetectResult(int cls_id, const char *name_str, float conf,
                                         int x1, int y1, int x2, int y2)
    {
        DetectionObject res;
        res.cls_id = cls_id;
        res.prop = conf;
        res.box = {x1, y1, x2, y2};
        strncpy(res.name, name_str, sizeof(res.name) - 1);
        res.name[sizeof(res.name) - 1] = '\0';
        return res;
    }
};

TEST_F(YoloV8SegPostProcessTest, run)
{
    std::vector<uint8_t> expected_seg_mask; // 改为vector<uint8_t>
    ASSERT_NO_THROW({
        expected_seg_mask = LoadSegmentationResult(path_seg_result);
    }) << "Failed to load expected segmentation mask";

    std::vector<DetectionObject> expected_results = {
        MakeDetectResult(5, "class_5", 0.9113f, 87, 137, 553, 439),
        MakeDetectResult(0, "class_0", 0.8998f, 108, 236, 227, 537),
        MakeDetectResult(0, "class_0", 0.8693f, 211, 241, 283, 508),
        MakeDetectResult(0, "class_0", 0.8655f, 477, 232, 559, 519),
        MakeDetectResult(0, "class_0", 0.5403f, 79, 327, 125, 514),
        MakeDetectResult(27, "class_27", 0.2741f, 248, 284, 259, 310)};

    std::vector<std::vector<int>> output_dims;
    std::vector<float> output_scales;
    std::vector<int32_t> output_zps;

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

    std::vector<void *> output_buffers;
    ASSERT_NO_THROW({
        output_buffers = LoadOutputBuffers(model_outputs_npz, 13);
    }) << "Failed to load output buffers from NPZ";

    // --- 调用被测方法 ---
    bool success = processor->run(&output_buffers, 640, 640,
                                  output_dims, output_scales, output_zps);

    const auto &result = processor->getResult();


    const auto &result_group = result.group;

    // --- 比较检测结果 ---
    EXPECT_TRUE(CompareDetectResultVectors(expected_results, result_group.results));

    // 在测试结束时比较分割掩码
    const auto& actual_results = processor->getResult().group;
    std::vector<std::vector<uint8_t>> expected_masks = {expected_seg_mask};
    
    // EXPECT_TRUE(CompareSegmentationMaskVectors(expected_masks, actual_results.segmentation_masks));
    std::string input_path = "apps/yolov8_seg_rknn/bus.jpg";
    cv::Mat orig_img = cv::imread(input_path, 1);
    cv::Mat result_img = orig_img.clone();
    processor->drawDetectionResults(result_img, result_group);
    std::string computed_out_path = "tmp/yolov8_seg_out.jpg";
    cv::imwrite(computed_out_path, result_img);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}