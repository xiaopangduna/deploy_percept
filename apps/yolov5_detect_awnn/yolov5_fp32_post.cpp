#include "yolov5_fp32_post.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <vector>

namespace yolov5_fp32_post
{

namespace
{

struct Object
{
    cv::Rect_<float> rect;
    int label{0};
    float prob{0.f};
};

static float sigmoid(float x)
{
    return 1.f / (1.f + std::exp(-x));
}

static float desigmoid(float x)
{
    return -std::log(1.f / x - 1.f);
}

static float intersectionArea(const Object &a, const Object &b)
{
    const cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsortDescentInplace(std::vector<Object> &objects, int left, int right)
{
    int i = left;
    int j = right;
    const float pivot = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > pivot)
        {
            ++i;
        }
        while (objects[j].prob < pivot)
        {
            --j;
        }
        if (i <= j)
        {
            std::swap(objects[i], objects[j]);
            ++i;
            --j;
        }
    }

    if (left < j)
    {
        qsortDescentInplace(objects, left, j);
    }
    if (i < right)
    {
        qsortDescentInplace(objects, i, right);
    }
}

static void nmsSortedBboxes(const std::vector<Object> &objects, std::vector<int> &picked, float nms_threshold)
{
    picked.clear();
    const int n = static_cast<int>(objects.size());
    std::vector<float> areas(n);
    for (int i = 0; i < n; ++i)
    {
        areas[i] = objects[i].rect.area();
    }

    for (int i = 0; i < n; ++i)
    {
        const Object &a = objects[i];
        int keep = 1;
        for (int j = 0; j < static_cast<int>(picked.size()); ++j)
        {
            const Object &b = objects[picked[j]];
            const float inter_area = intersectionArea(a, b);
            const float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (union_area > 0.f && inter_area / union_area > nms_threshold)
            {
                keep = 0;
                break;
            }
        }
        if (keep)
        {
            picked.push_back(i);
        }
    }
}

static void generateProposals(
    int stride,
    const float *feat,
    float prob_threshold,
    std::vector<Object> &objects,
    int letterbox_cols,
    int letterbox_rows)
{
    static const float anchors[18] = {
        10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326};

    const int anchor_num = 3;
    const int feat_w = letterbox_cols / stride;
    const int feat_h = letterbox_rows / stride;
    const int cls_num = 80;

    int anchor_group = 1;
    if (stride == 16)
    {
        anchor_group = 2;
    }
    if (stride == 32)
    {
        anchor_group = 3;
    }

    const float deprob_threshold = desigmoid(prob_threshold);
    const int feat_size = feat_w * feat_h;
    const int feat_size_cls_5 = feat_size * (cls_num + 5);

    for (int h = 0; h < feat_h; ++h)
    {
        const int h_feat_w_cls_5 = h * feat_w * (cls_num + 5);
        for (int w = 0; w < feat_w; ++w)
        {
            const int w_cls_5 = w * (cls_num + 5);
            for (int a = 0; a < anchor_num; ++a)
            {
                int class_index = 0;
                float class_score = -FLT_MAX;
                const int a_idx = a * feat_size_cls_5 + h_feat_w_cls_5 + w_cls_5;
                const float *feat_ptr = &feat[a_idx + 4];
                for (int s = 0; s < cls_num; ++s)
                {
                    if (*(feat_ptr + s + 1) > class_score)
                    {
                        class_index = s;
                        class_score = *(feat_ptr + s + 1);
                    }
                }

                const float box_score = *feat_ptr;
                float final_score = 0.f;
                if (box_score >= deprob_threshold && class_score >= deprob_threshold)
                {
                    final_score = sigmoid(box_score) * sigmoid(class_score);
                }

                if (final_score >= prob_threshold)
                {
                    const int loc_idx = a_idx;
                    const float dx = sigmoid(feat[loc_idx + 0]);
                    const float dy = sigmoid(feat[loc_idx + 1]);
                    const float dw = sigmoid(feat[loc_idx + 2]);
                    const float dh = sigmoid(feat[loc_idx + 3]);
                    const float pred_cx = (dx * 2.f - 0.5f + w) * stride;
                    const float pred_cy = (dy * 2.f - 0.5f + h) * stride;
                    const float anchor_w = anchors[(anchor_group - 1) * 6 + a * 2 + 0];
                    const float anchor_h = anchors[(anchor_group - 1) * 6 + a * 2 + 1];
                    const float pred_w = dw * dw * 4.f * anchor_w;
                    const float pred_h = dh * dh * 4.f * anchor_h;
                    const float x0 = pred_cx - pred_w * 0.5f;
                    const float y0 = pred_cy - pred_h * 0.5f;
                    const float x1 = pred_cx + pred_w * 0.5f;
                    const float y1 = pred_cy + pred_h * 0.5f;

                    Object obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0;
                    obj.rect.height = y1 - y0;
                    obj.label = class_index;
                    obj.prob = final_score;
                    objects.push_back(obj);
                }
            }
        }
    }
}

} // namespace

int run(
    float *outputs[3],
    int letterbox_w,
    int letterbox_h,
    std::vector<Detection> &detections,
    float conf_threshold,
    float nms_threshold)
{
    detections.clear();

    std::vector<Object> proposals;
    std::vector<Object> objects8;
    std::vector<Object> objects16;
    std::vector<Object> objects32;

    generateProposals(32, outputs[2], conf_threshold, objects32, letterbox_w, letterbox_h);
    proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    generateProposals(16, outputs[1], conf_threshold, objects16, letterbox_w, letterbox_h);
    proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    generateProposals(8, outputs[0], conf_threshold, objects8, letterbox_w, letterbox_h);
    proposals.insert(proposals.end(), objects8.begin(), objects8.end());

    if (proposals.empty())
    {
        return 0;
    }

    qsortDescentInplace(proposals, 0, static_cast<int>(proposals.size()) - 1);
    std::vector<int> picked;
    nmsSortedBboxes(proposals, picked, nms_threshold);

    detections.reserve(picked.size());
    for (const int idx : picked)
    {
        const Object &obj = proposals[idx];
        Detection det;
        det.x0 = obj.rect.x;
        det.y0 = obj.rect.y;
        det.x1 = obj.rect.x + obj.rect.width;
        det.y1 = obj.rect.y + obj.rect.height;
        det.label = obj.label;
        det.score = obj.prob;
        detections.push_back(det);
    }

    return static_cast<int>(detections.size());
}

void draw(cv::Mat &bgr, const std::vector<Detection> &detections)
{
    static const char *class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"};

    for (const Detection &det : detections)
    {
        const cv::Rect rect(
            static_cast<int>(det.x0),
            static_cast<int>(det.y0),
            static_cast<int>(det.x1 - det.x0),
            static_cast<int>(det.y1 - det.y0));

        std::fprintf(
            stderr,
            "%2d: %3.0f%%, [%4.0f, %4.0f, %4.0f, %4.0f], %s\n",
            det.label,
            det.score * 100.f,
            det.x0,
            det.y0,
            det.x1,
            det.y1,
            class_names[det.label]);

        cv::rectangle(bgr, rect, cv::Scalar(255, 0, 0), 2);

        char text[256];
        std::snprintf(text, sizeof(text), "%s %.1f%%", class_names[det.label], det.score * 100.f);

        int baseline = 0;
        const cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        int x = rect.x;
        int y = rect.y - label_size.height - baseline;
        if (y < 0)
        {
            y = 0;
        }
        if (x + label_size.width > bgr.cols)
        {
            x = bgr.cols - label_size.width;
        }

        cv::rectangle(
            bgr,
            cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseline)),
            cv::Scalar(255, 255, 255),
            -1);
        cv::putText(
            bgr,
            text,
            cv::Point(x, y + label_size.height),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 0, 0),
            1);
    }
}

} // namespace yolov5_fp32_post
