#pragma once

#include "deploy_percept/post_process/BasePostProcess.hpp"
#include "deploy_percept/post_process/types.hpp"
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

namespace deploy_percept {
namespace post_process {

class YoloBasePostProcess : public BasePostProcess {
public:
    YoloBasePostProcess();
    virtual ~YoloBasePostProcess() = default;

    /**
     * @brief 将浮点数值钳位到指定整数范围内
     * @param val 输入的浮点数值
     * @param min 最小边界值
     * @param max 最大边界边界值
     * @return 钳位后的整数值
     * @details 如果val小于min则返回min，如果val大于max则返回max，否则返回val的整数部分
     */
    static int clamp(float val, int min, int max);

    /**
     * @brief 将32位浮点数量化为8位整数
     * @param f32 输入的32位浮点数
     * @param zp 量化零点值
     * @param scale 量化缩放因子
     * @return 量化后的8位整数(-128到127范围)
     * @details 使用公式: qnt = round(f32 / scale + zp)，并钳位到int8范围
     */
    static int8_t qntF32ToAffine(float f32, int32_t zp, float scale);

    /**
     * @brief 将8位整数量化值反量化为32位浮点数
     * @param qnt 量化后的8位整数值
     * @param zp 量化零点值
     * @param scale 量化缩放因子
     * @return 反量化后的32位浮点数
     * @details 使用公式: f32 = (qnt - zp) * scale
     */
    static float deqntAffineToF32(int8_t qnt, int32_t zp, float scale);

    /**
     * @brief 将浮点数值钳位到指定浮点数范围内
     * @param val 输入的浮点数值
     * @param min 最小边界值
     * @param max 最大边界值
     * @return 钳位后的32位整数值
     * @details 如果val小于min则返回min，如果val大于max则返回max，否则返回val
     */
    static int32_t clip(float val, float min, float max);

    /**
     * @brief 对索引数组进行快速排序(降序)
     * @param input 待排序的浮点数数组
     * @param left 排序起始索引
     * @param right 排序结束索引
     * @param indices 与input对应的索引数组
     * @return 排序完成后基准元素的最终位置
     * @details 使用快速排序算法对input数组进行降序排序，同时调整indices数组
     *          主要用于NMS算法前对检测框置信度进行排序
     */
    static int quickSortIndices(std::vector<float> &input, int left, int right, std::vector<int> &indices);

    /**
     * @brief 执行非极大值抑制(NMS)算法
     * @param validCount 有效的检测框数量
     * @param outputLocations 检测框坐标数组[x,y,w,h]格式
     * @param classIds 类别ID数组
     * @param order 排序后的索引数组
     * @param filterId 要过滤的类别ID
     * @param threshold IOU阈值
     * @return 执行状态，0表示成功
     * @details 通过计算IOU去除重叠度高的冗余检测框，保留置信度最高的检测结果
     */
    static int retainHighestScoringBoxesByNMS(int validCount, std::vector<float>& outputLocations, std::vector<int> classIds, 
            std::vector<int>& order, int filterId, float threshold);

    /**
     * @brief 计算两个边界框的交并比(IOU)
     * @param xmin0 第一个框的最小x坐标
     * @param ymin0 第一个框的最小y坐标
     * @param xmax0 第一个框的最大x坐标
     * @param ymax0 第一个框的最大y坐标
     * @param xmin1 第二个框的最小x坐标
     * @param ymin1 第二个框的最小y坐标
     * @param xmax1 第二个框的最大x坐标
     * @param ymax1 第二个框的最大y坐标
     * @return 两个边界框的交并比值[0,1]
     * @details 使用标准IOU计算公式:(交集面积)/(并集面积)
     */
    static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, 
                                float xmin1, float ymin1, float xmax1, float ymax1);

    /**
     * @brief 计算分割掩码
     * @param A 分割特征向量
     * @param B 原型掩码
     * @param C 输出掩码
     * @param ROWS_A A的行数
     * @param COLS_A A的列数
     * @param COLS_B B的列数
     * @details 通过矩阵乘法计算分割掩码
     */
    static void computeSegMask(std::vector<float> &A, float *B, uint8_t *C, int ROWS_A, int COLS_A, int COLS_B);

    /**
     * @brief 调整分割掩码大小
     * @param input_image 输入掩码图像
     * @param input_width 输入宽度
     * @param input_height 输入高度
     * @param boxes_num 检测框数量
     * @param output_image 输出掩码图像
     * @param target_width 目标宽度
     * @param target_height 目标高度
     * @details 将分割掩码调整为指定尺寸
     */
    static void resizeSegMasks(uint8_t *input_image, int input_width, int input_height, int boxes_num,
                              uint8_t *output_image, int target_width, int target_height);

    /**
     * @brief 合并边界框和掩码
     * @param seg_mask 分割掩码
     * @param all_mask_in_one 合并后的掩码
     * @param boxes 边界框坐标
     * @param boxes_num 检测框数量
     * @param cls_id 类别ID
     * @param height 图像高度
     * @param width 图像宽度
     * @details 将各个检测框的分割掩码合并成一个完整的分割掩码
     */
    static void mergeBoxMasks(uint8_t *seg_mask, uint8_t *all_mask_in_one, float *boxes, int boxes_num,
                             int *cls_id, int height, int width);

    /**
     * @brief 反转分割掩码
     * @param seg_mask 分割掩码
     * @param cropped_seg 裁剪后的分割掩码
     * @param seg_mask_real 真实分割掩码
     * @param input_image_height 输入图像高度
     * @param input_image_width 输入图像宽度
     * @param cropped_height 裁剪高度
     * @param cropped_width 裁剪宽度
     * @param ori_in_height 原始输入高度
     * @param ori_in_width 原始输入宽度
     * @param y_pad y轴填充
     * @param x_pad x轴填充
     * @details 将裁剪后的分割掩码映射回原始图像尺寸
     */
    static void seg_reverse(uint8_t *seg_mask, uint8_t *cropped_seg, uint8_t *seg_mask_real,
                           int input_image_height, int input_image_width, int cropped_height, int cropped_width,
                           int ori_in_height, int ori_in_width, int y_pad, int x_pad);

    /**
     * @brief 绘制检测和分割结果
     * @param image 要绘制的图像
     * @param results 检测结果
     * @details 在图像上绘制检测框和分割掩码
     */
    void drawDetectionResults(cv::Mat &image, const ResultGroup &results) const;

protected:
    /**
     * @brief 收集NMS后的检测结果
     * @param indexArray 索引数组
     * @param filterBoxes 过滤后的边界框
     * @param classId 类别ID
     * @param objProbs 置信度
     * @param filterSegments 过滤后的分割特征
     * @param validCount 有效数量
     * @param filterSegments_by_nms NMS后的分割特征
     * @param last_count 最终计数
     * @param input_image_width 输入图像宽度（仅YOLOv8使用）
     * @param input_image_height 输入图像高度（仅YOLOv8使用）
     * @details 在NMS之后收集检测结果
     */
    void collectDetectionsAfterNMS(
        const std::vector<int> &indexArray,
        const std::vector<float> &filterBoxes,
        const std::vector<int> &classId,
        const std::vector<float> &objProbs,
        const std::vector<float> &filterSegments,
        int validCount,
        std::vector<float> &filterSegments_by_nms,
        int &last_count,
        int input_image_width = 0,
        int input_image_height = 0);
};

} // namespace post_process
} // namespace deploy_percept