#pragma once

#include "deploy_percept/post_process/BasePostProcess.hpp"
#include "deploy_percept/post_process/types.hpp"
#include <vector>
#include <string>

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
     * @param max 最大边界值
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
     * @details 使用快速排序算法对input数组进行降序排序，同时同步调整indices数组
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
    static int nms(int validCount, std::vector<float>& outputLocations, std::vector<int> classIds, 
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
};

} // namespace post_process
} // namespace deploy_percept
