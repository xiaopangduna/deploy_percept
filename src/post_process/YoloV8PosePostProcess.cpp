#include "deploy_percept/post_process/YoloV8PosePostProcess.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <numeric>
#include <set>

namespace deploy_percept
{
namespace post_process
{

namespace
{

float sigmoid(float x)
{
    return 1.f / (1.f + std::exp(-x));
}

float unsigmoid(float y)
{
    return -std::log((1.f / y) - 1.f);
}

void softmax(float *input, int size)
{
    float max_val = input[0];
    for (int i = 1; i < size; ++i)
    {
        if (input[i] > max_val)
        {
            max_val = input[i];
        }
    }
    float sum_exp = 0.f;
    for (int i = 0; i < size; ++i)
    {
        sum_exp += std::exp(input[i] - max_val);
    }
    for (int i = 0; i < size; ++i)
    {
        input[i] = std::exp(input[i] - max_val) / sum_exp;
    }
}

/** IEEE-754 half to float (RKNN float16 输出) */
float fp16BitsToFloat(uint16_t h)
{
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = static_cast<uint32_t>(h & 0x03ffu);
    uint32_t u;
    if (exp == 0)
    {
        if (mant == 0)
        {
            u = sign;
        }
        else
        {
            while ((mant & 0x0400u) == 0u)
            {
                mant <<= 1u;
                exp--;
            }
            exp++;
            mant &= 0x03ffu;
            u = sign | (((exp + (127u - 15u)) & 0xffu) << 23) | (mant << 13);
        }
    }
    else if (exp == 31u)
    {
        u = sign | 0x7f800000u | (mant << 13);
    }
    else
    {
        u = sign | (((exp + (127u - 15u)) & 0xffu) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &u, sizeof(out));
    return out;
}

static float deqntU8ToF32(uint8_t qnt, int32_t zp, float scale)
{
    return (static_cast<float>(qnt) - static_cast<float>(zp)) * scale;
}

/** 与 third_party/rknpu2/include/rknn_api.h rknn_tensor_type 一致 */
enum RknnTensorTypeCompat : int32_t
{
    kFloat32 = 0,
    kFloat16 = 1,
    kInt8 = 2,
    kUint8 = 3,
};

} // namespace

YoloV8PosePostProcess::YoloV8PosePostProcess(const Params &params) : params_(params) {}

int YoloV8PosePostProcess::decodeHeadInt8(const int8_t *input,
                                         int grid_h,
                                         int grid_w,
                                         int stride,
                                         int index_base,
                                         std::vector<float> &boxes,
                                         std::vector<float> &box_scores,
                                         std::vector<int> &class_id,
                                         float conf_threshold,
                                         int32_t zp,
                                         float scale,
                                         int obj_class_num)
{
    const int input_loc_len = 64;
    const int tensor_len = input_loc_len + obj_class_num;
    int valid_count = 0;
    const int8_t thres_i8 = YoloBasePostProcess::qntF32ToAffine(unsigmoid(conf_threshold), zp, scale);

    for (int h = 0; h < grid_h; ++h)
    {
        for (int w = 0; w < grid_w; ++w)
        {
            for (int a = 0; a < obj_class_num; ++a)
            {
                const int idx = (input_loc_len + a) * grid_w * grid_h + h * grid_w + w;
                if (input[idx] < thres_i8)
                {
                    continue;
                }
                const float box_conf_f32 =
                    sigmoid(YoloBasePostProcess::deqntAffineToF32(input[idx], zp, scale));
                float loc[input_loc_len];
                for (int i = 0; i < input_loc_len; ++i)
                {
                    loc[i] = YoloBasePostProcess::deqntAffineToF32(
                        input[i * grid_w * grid_h + h * grid_w + w], zp, scale);
                }
                for (int i = 0; i < input_loc_len / 16; ++i)
                {
                    softmax(&loc[i * 16], 16);
                }
                float xywh_[4] = {0, 0, 0, 0};
                float xywh[4] = {0, 0, 0, 0};
                for (int dfl = 0; dfl < 16; ++dfl)
                {
                    xywh_[0] += loc[dfl] * static_cast<float>(dfl);
                    xywh_[1] += loc[1 * 16 + dfl] * static_cast<float>(dfl);
                    xywh_[2] += loc[2 * 16 + dfl] * static_cast<float>(dfl);
                    xywh_[3] += loc[3 * 16 + dfl] * static_cast<float>(dfl);
                }
                xywh_[0] = (static_cast<float>(w) + 0.5f) - xywh_[0];
                xywh_[1] = (static_cast<float>(h) + 0.5f) - xywh_[1];
                xywh_[2] = (static_cast<float>(w) + 0.5f) + xywh_[2];
                xywh_[3] = (static_cast<float>(h) + 0.5f) + xywh_[3];
                xywh[0] = ((xywh_[0] + xywh_[2]) / 2.f) * static_cast<float>(stride);
                xywh[1] = ((xywh_[1] + xywh_[3]) / 2.f) * static_cast<float>(stride);
                xywh[2] = (xywh_[2] - xywh_[0]) * static_cast<float>(stride);
                xywh[3] = (xywh_[3] - xywh_[1]) * static_cast<float>(stride);
                xywh[0] = xywh[0] - xywh[2] / 2.f;
                xywh[1] = xywh[1] - xywh[3] / 2.f;

                boxes.push_back(xywh[0]);
                boxes.push_back(xywh[1]);
                boxes.push_back(xywh[2]);
                boxes.push_back(xywh[3]);
                boxes.push_back(static_cast<float>(index_base + h * grid_w + w));
                box_scores.push_back(box_conf_f32);
                class_id.push_back(a);
                valid_count++;
            }
        }
    }
    return valid_count;
}

int YoloV8PosePostProcess::nmsPose(int valid_count,
                                   std::vector<float> &output_locations,
                                   std::vector<int> &class_ids,
                                   std::vector<int> &order,
                                   int filter_id,
                                   float threshold)
{
    for (int i = 0; i < valid_count; ++i)
    {
        int n = order[i];
        if (n == -1 || class_ids[n] != filter_id)
        {
            continue;
        }
        for (int j = i + 1; j < valid_count; ++j)
        {
            int m = order[j];
            if (m == -1 || class_ids[m] != filter_id)
            {
                continue;
            }
            const float xmin0 = output_locations[n * 5 + 0];
            const float ymin0 = output_locations[n * 5 + 1];
            const float xmax0 = output_locations[n * 5 + 0] + output_locations[n * 5 + 2];
            const float ymax0 = output_locations[n * 5 + 1] + output_locations[n * 5 + 3];

            const float xmin1 = output_locations[m * 5 + 0];
            const float ymin1 = output_locations[m * 5 + 1];
            const float xmax1 = output_locations[m * 5 + 0] + output_locations[m * 5 + 2];
            const float ymax1 = output_locations[m * 5 + 1] + output_locations[m * 5 + 3];

            const float iou = YoloBasePostProcess::CalculateOverlap(
                xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);
            if (iou > threshold)
            {
                order[j] = -1;
            }
        }
    }
    return 0;
}

float YoloV8PosePostProcess::readKeypointValue(const void *buf,
                                               int kp_index,
                                               int keypoint_j,
                                               int plane,
                                               int32_t tensor_type,
                                               int32_t zp,
                                               float scale)
{
    const int L = Params::kKeypointAnchorLen;
    const size_t offset = static_cast<size_t>(keypoint_j) * 3u * static_cast<size_t>(L) +
                          static_cast<size_t>(plane) * static_cast<size_t>(L) + static_cast<size_t>(kp_index);
    const auto *b8 = static_cast<const uint8_t *>(buf);

    switch (tensor_type)
    {
    case kFloat16: {
        uint16_t h;
        std::memcpy(&h, b8 + offset * sizeof(uint16_t), sizeof(h));
        return fp16BitsToFloat(h);
    }
    case kFloat32:
        return *reinterpret_cast<const float *>(b8 + offset * sizeof(float));
    case kInt8:
        return YoloBasePostProcess::deqntAffineToF32(
            *reinterpret_cast<const int8_t *>(b8 + offset * sizeof(int8_t)), zp, scale);
    case kUint8:
        return deqntU8ToF32(*reinterpret_cast<const uint8_t *>(b8 + offset * sizeof(uint8_t)), zp, scale);
    default:
        return YoloBasePostProcess::deqntAffineToF32(
            *reinterpret_cast<const int8_t *>(b8 + offset * sizeof(int8_t)), zp, scale);
    }
}

void YoloV8PosePostProcess::gatherKeypointsForDetection(PoseDetectionObject &out,
                                                        const void *kpt_buf,
                                                        int32_t kpt_type,
                                                        int32_t kpt_zp,
                                                        float kpt_scale,
                                                        int keypoints_index,
                                                        int model_in_w,
                                                        int model_in_h,
                                                        float letter_scale,
                                                        float x_pad,
                                                        float y_pad)
{
    for (int j = 0; j < 17; ++j)
    {
        const float x =
            readKeypointValue(kpt_buf, keypoints_index, j, 0, kpt_type, kpt_zp, kpt_scale);
        const float y =
            readKeypointValue(kpt_buf, keypoints_index, j, 1, kpt_type, kpt_zp, kpt_scale);
        const float c =
            readKeypointValue(kpt_buf, keypoints_index, j, 2, kpt_type, kpt_zp, kpt_scale);
        out.keypoints[j][0] = (x - x_pad) / letter_scale;
        out.keypoints[j][1] = (y - y_pad) / letter_scale;
        out.keypoints[j][2] = c;
    }
    (void)model_in_w;
    (void)model_in_h;
}

bool YoloV8PosePostProcess::run(const std::vector<void *> &outputs,
                                int input_image_width,
                                int input_image_height,
                                std::vector<std::vector<int>> &output_dims,
                                std::vector<float> &output_scales,
                                std::vector<int32_t> &output_zps,
                                const std::vector<int32_t> &output_types)
{
    result_.group.objects.clear();
    result_.group.count = 0;

    if (outputs.size() < 4 || output_dims.size() < 4 || output_scales.size() < 4 ||
        output_zps.size() < 4 || output_types.size() < 4)
    {
        return false;
    }

    std::vector<float> filter_boxes;
    std::vector<float> obj_probs;
    std::vector<int> class_ids;
    int valid_count = 0;
    int index = 0;

    const int model_in_h = input_image_height;
    const int model_in_w = input_image_width;

    for (int i = 0; i < 3; ++i)
    {
        const int grid_h = output_dims[i][2];
        const int grid_w = output_dims[i][3];
        const int stride = model_in_h / grid_h;
        const auto *buf = static_cast<const int8_t *>(outputs[i]);
        valid_count += decodeHeadInt8(buf,
                                      grid_h,
                                      grid_w,
                                      stride,
                                      index,
                                      filter_boxes,
                                      obj_probs,
                                      class_ids,
                                      params_.conf_threshold,
                                      output_zps[i],
                                      output_scales[i],
                                      params_.obj_class_num);
        index += grid_h * grid_w;
    }

    if (valid_count <= 0)
    {
        return true;
    }

    std::vector<int> order(valid_count);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&obj_probs](int a, int b) {
        return obj_probs[a] > obj_probs[b];
    });

    std::set<int> class_set(class_ids.begin(), class_ids.end());
    for (int c : class_set)
    {
        nmsPose(valid_count, filter_boxes, class_ids, order, c, params_.nms_threshold);
    }

    const void *kpt_buf = outputs[3];
    const int32_t kpt_type = output_types[3];
    const int32_t kpt_zp = output_zps[3];
    const float kpt_scale = output_scales[3];

    /** 无 LetterBox：与原图一致时 scale=1，pad=0 */
    const float letter_scale = 1.f;
    const float x_pad = 0.f;
    const float y_pad = 0.f;

    int last_count = 0;
    for (int i = 0; i < valid_count; ++i)
    {
        if (order[i] == -1 || last_count >= params_.obj_numb_max_size)
        {
            continue;
        }
        const int n = order[i];
        const float x1 = filter_boxes[n * 5 + 0] - x_pad;
        const float y1 = filter_boxes[n * 5 + 1] - y_pad;
        const float w = filter_boxes[n * 5 + 2];
        const float h = filter_boxes[n * 5 + 3];
        const int keypoints_index = static_cast<int>(filter_boxes[n * 5 + 4]);

        PoseDetectionObject det{};
        gatherKeypointsForDetection(det,
                                    kpt_buf,
                                    kpt_type,
                                    kpt_zp,
                                    kpt_scale,
                                    keypoints_index,
                                    model_in_w,
                                    model_in_h,
                                    letter_scale,
                                    x_pad,
                                    y_pad);

        det.cls_id = class_ids[n];
        det.prop = obj_probs[n];
        det.box.left = YoloBasePostProcess::clamp(x1, 0, model_in_w) / letter_scale;
        det.box.top = YoloBasePostProcess::clamp(y1, 0, model_in_h) / letter_scale;
        det.box.right = YoloBasePostProcess::clamp(x1 + w, 0, model_in_w) / letter_scale;
        det.box.bottom = YoloBasePostProcess::clamp(y1 + h, 0, model_in_h) / letter_scale;

        result_.group.objects.push_back(det);
        last_count++;
    }

    result_.group.count = last_count;
    return true;
}

void YoloV8PosePostProcess::drawPoseResults(cv::Mat &image, const PoseResultGroup &results) const
{
    static const int skeleton[38] = {16, 14, 14, 12, 17, 15, 15, 13, 12, 13, 6, 12, 7, 13, 6, 7, 6, 8,
                                     7, 9, 8, 10, 9, 11, 2, 3, 1, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7};

    const cv::Scalar color_box(255, 0, 0);
    const cv::Scalar color_line(0, 165, 255);
    const cv::Scalar color_kp(0, 255, 255);
    const float kpt_conf_thr = 0.25f; // 关键点置信度阈值：太低的不画

    for (size_t i = 0; i < results.objects.size(); ++i)
    {
        const PoseDetectionObject &det = results.objects[i];
        cv::rectangle(image,
                      cv::Point(det.box.left, det.box.top),
                      cv::Point(det.box.right, det.box.bottom),
                      color_box,
                      2);

        // 类别名/ID + 置信度
        std::string cls_name = (det.cls_id == 0) ? "人" : ("类" + std::to_string(det.cls_id));
        const float prop_pct = det.prop * 100.f;
        std::string label = cls_name + " id=" + std::to_string(det.cls_id) + " " + std::to_string(prop_pct) + "%";
        int baseline = 0;
        const cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);

        const int x0 = std::max(0, det.box.left);
        const int y0 = std::max(0, det.box.top);
        const int y_text_bottom = std::max(0, y0 - 5);
        const int y_text_top = std::max(0, y_text_bottom - textSize.height - baseline);

        cv::rectangle(image,
                      cv::Point(x0, y_text_top),
                      cv::Point(std::min(image.cols - 1, x0 + textSize.width), y_text_bottom),
                      color_box,
                      -1);
        cv::putText(image,
                    label,
                    cv::Point(x0, y_text_bottom),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    cv::Scalar(255, 255, 255),
                    1);

        for (int j = 0; j < 38 / 2; ++j)
        {
            const int a = skeleton[2 * j] - 1;
            const int b = skeleton[2 * j + 1] - 1;
            if (det.keypoints[a][2] >= kpt_conf_thr && det.keypoints[b][2] >= kpt_conf_thr)
            {
                cv::line(image,
                         cv::Point(static_cast<int>(det.keypoints[a][0]), static_cast<int>(det.keypoints[a][1])),
                         cv::Point(static_cast<int>(det.keypoints[b][0]), static_cast<int>(det.keypoints[b][1])),
                         color_line,
                         2);
            }
        }
        for (int j = 0; j < 17; ++j)
        {
            if (det.keypoints[j][2] >= kpt_conf_thr)
            {
                cv::circle(image,
                           cv::Point(static_cast<int>(det.keypoints[j][0]), static_cast<int>(det.keypoints[j][1])),
                           3,
                           color_kp,
                           -1);
            }
        }
    }
}

} // namespace post_process
} // namespace deploy_percept
