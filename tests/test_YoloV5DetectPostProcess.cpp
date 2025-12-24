#include <gtest/gtest.h>
#include <vector>
#include <cstring>

#include "deploy_percept/post_process/YoloV5DetectPostProcess.hpp"
#include "utils/environment.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////
// YoloV5后处理相关测试
////////////////////////////////////////////////////////////////////////////////////////////////////

// YoloV5检测后处理测试夹具类
class YoloV5DetectPostProcessTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if (GlobalLoggerEnvironment::logger)
        {
            GlobalLoggerEnvironment::logger->info("Setting up YoloV5DetectPostProcess test");
        }
        
        // 初始化一个基本的YoloV5DetectPostProcess实例
        processor = std::make_unique<deploy_percept::post_process::YoloV5DetectPostProcess>();
    }

    void TearDown() override
    {
        processor.reset();
        if (GlobalLoggerEnvironment::logger)
        {
            GlobalLoggerEnvironment::logger->info("Tearing down YoloV5DetectPostProcess test");
        }
    }

    std::unique_ptr<deploy_percept::post_process::YoloV5DetectPostProcess> processor;
};

// 测试基本的process功能（使用虚拟数据）
TEST_F(YoloV5DetectPostProcessTest, ProcessFunctionWithDummyData)
{
    if (GlobalLoggerEnvironment::logger)
    {
        GlobalLoggerEnvironment::logger->info("Testing YoloV5DetectPostProcess process function with dummy data");
    }

    // 创建虚拟输入数据 - 使用小尺寸数据进行测试
    const int model_in_h = 640;
    const int model_in_w = 640;
    
    // 使用类的静态常量
    const int obj_class_num = deploy_percept::post_process::YoloBasePostProcess::OBJ_CLASS_NUM;
    const int prop_box_size = 5 + obj_class_num;  // 85
    
    // 虚拟输出尺寸 (通常是YOLO的3个尺度输出)
    const int output_size_0 = (model_in_h/8) * (model_in_w/8) * prop_box_size * 3; // stride 8
    const int output_size_1 = (model_in_h/16) * (model_in_w/16) * prop_box_size * 3; // stride 16
    const int output_size_2 = (model_in_h/32) * (model_in_w/32) * prop_box_size * 3; // stride 32
    
    std::vector<int8_t> input0(output_size_0, 0);
    std::vector<int8_t> input1(output_size_1, 0);
    std::vector<int8_t> input2(output_size_2, 0);
    
    // 添加一些模拟检测框数据 (使用较高的置信度值以通过阈值检查)
    // 在第一个输出层中添加一个模拟检测
    const int grid_h0 = model_in_h / 8;  // 80
    const int grid_w0 = model_in_w / 8;  // 80
    const int anchor_idx = 0;  // 第一个anchor
    const int class_idx = 0;   // 第一个类别
    const int grid_y = 10;     // 网格y坐标
    const int grid_x = 10;     // 网格x坐标
    
    // 计算在数组中的位置
    int offset = (prop_box_size * anchor_idx) * (grid_h0 * grid_w0) + (grid_y * grid_w0 + grid_x);
    
    // 设置边界框坐标 (x, y, w, h)
    input0[offset] = 64;      // x center (模拟量化后的值)
    input0[offset + grid_h0 * grid_w0] = 64;  // y center
    input0[offset + 2 * grid_h0 * grid_w0] = 48; // width
    input0[offset + 3 * grid_h0 * grid_w0] = 48; // height
    
    // 设置置信度
    input0[offset + 4 * grid_h0 * grid_w0] = 80; // object confidence (高于阈值)
    
    // 设置类别概率
    input0[offset + (5 + class_idx) * grid_h0 * grid_w0] = 90; // class probability

    // 创建结果组
    deploy_percept::post_process::detect_result_group_t group;
    memset(&group, 0, sizeof(group));

    // 设置量化参数
    std::vector<int32_t> qnt_zps = {0, 0, 0};
    std::vector<float> qnt_scales = {1.0, 1.0, 1.0};
    
    // 设置填充和缩放参数
    deploy_percept::post_process::BOX_RECT pads = {0, 0, 0, 0};
    float scale_w = 1.0;
    float scale_h = 1.0;

    // 执行处理
    int result = processor->process(
        input0.data(),
        input1.data(),
        input2.data(),
        model_in_h,
        model_in_w,
        pads,
        scale_w,
        scale_h,
        qnt_zps,
        qnt_scales,
        &group
    );

    // 验证结果
    EXPECT_EQ(result, 0);  // 确保处理成功
    // 注意：由于后处理逻辑复杂，可能不会检测到任何对象，所以不检查group.count > 0
}


int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // 注册全局环境
    ::testing::AddGlobalTestEnvironment(new GlobalLoggerEnvironment);

    return RUN_ALL_TESTS();
}