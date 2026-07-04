#include "deploy_percept/utils/vis_draw.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cstdio>
#include <string>

namespace deploy_percept
{
namespace utils
{

namespace
{

std::string formatScorePercent(const float score)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.2f%%", score * 100.f);
    return buf;
}

std::string formatObjectCaption(const int class_id,
                                const float score,
                                const data::ClassLabelTable *labels)
{
    const std::string score_str = formatScorePercent(score);
    if (labels != nullptr)
    {
        const char *name = data::lookup_label(*labels, class_id);
        if (name != nullptr)
        {
            return std::string(name) + " | " + score_str;
        }
    }
    return "id " + std::to_string(class_id) + " | " + score_str;
}

std::string formatDetectionCaption(const Detection &det, const data::ClassLabelTable *labels)
{
    return formatObjectCaption(det.class_id, det.score, labels);
}

void drawDetectionBoxes(cv::Mat &image,
                        const std::vector<Detection> &detections,
                        const data::ClassLabelTable *labels)
{
    unsigned char class_colors[][3] = {
        {255, 56, 56},
        {255, 157, 151},
        {255, 112, 31},
        {255, 178, 29},
        {207, 210, 49},
        {72, 249, 10},
        {146, 204, 23},
        {61, 219, 134},
        {26, 147, 52},
        {0, 212, 187},
        {44, 153, 168},
        {0, 194, 255},
        {52, 69, 147},
        {100, 115, 255},
        {0, 24, 236},
        {132, 56, 255},
        {82, 0, 133},
        {203, 56, 255},
        {255, 149, 200},
        {255, 55, 199}};

    for (const Detection &det : detections)
    {
        const cv::Scalar color = cv::Scalar(class_colors[det.class_id % 20][2],
                                            class_colors[det.class_id % 20][1],
                                            class_colors[det.class_id % 20][0]);

        cv::rectangle(image,
                      cv::Point(det.box.left, det.box.top),
                      cv::Point(det.box.right, det.box.bottom),
                      color,
                      2);

        const std::string label = formatDetectionCaption(det, labels);
        int baseline = 0;
        const cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(image,
                      cv::Point(det.box.left, det.box.top - textSize.height - 10),
                      cv::Point(det.box.left + textSize.width, det.box.top),
                      color,
                      -1);
        cv::putText(image,
                    label,
                    cv::Point(det.box.left, det.box.top - 5),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(255, 255, 255),
                    1);
    }
}

void drawLabelMaskOverlay(cv::Mat &image, const std::vector<std::uint8_t> &label_mask)
{
    if (label_mask.empty())
    {
        return;
    }

    unsigned char class_colors[][3] = {
        {255, 56, 56},
        {255, 157, 151},
        {255, 112, 31},
        {255, 178, 29},
        {207, 210, 49},
        {72, 249, 10},
        {146, 204, 23},
        {61, 219, 134},
        {26, 147, 52},
        {0, 212, 187},
        {44, 153, 168},
        {0, 194, 255},
        {52, 69, 147},
        {100, 115, 255},
        {0, 24, 236},
        {132, 56, 255},
        {82, 0, 133},
        {203, 56, 255},
        {255, 149, 200},
        {255, 55, 199}};

    const int width = image.cols;
    const int height = image.rows;
    const float alpha = 0.5f;

    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            const int mask_value = label_mask[static_cast<std::size_t>(h * width + w)];

            if (mask_value != 0)
            {
                const cv::Vec3b color = cv::Vec3b(class_colors[mask_value % 20][0],
                                                  class_colors[mask_value % 20][1],
                                                  class_colors[mask_value % 20][2]);

                cv::Vec3b &pixel = image.at<cv::Vec3b>(h, w);

                pixel[0] = static_cast<unsigned char>(color[0] * (1 - alpha) + pixel[0] * alpha);
                pixel[1] = static_cast<unsigned char>(color[1] * (1 - alpha) + pixel[1] * alpha);
                pixel[2] = static_cast<unsigned char>(color[2] * (1 - alpha) + pixel[2] * alpha);
            }
        }
    }
}

std::string formatPoseCaption(const PoseDetection &det, const data::ClassLabelTable *labels)
{
    return formatObjectCaption(det.class_id, det.score, labels);
}

} // namespace

void drawDetectionResults(cv::Mat &image,
                          const DetectionResult &results,
                          const data::ClassLabelTable *labels)
{
    if (image.empty())
    {
        return;
    }

    drawDetectionBoxes(image, results.detections, labels);
}

void drawSegmentationResults(cv::Mat &image,
                             const SegmentationResult &results,
                             const data::ClassLabelTable *labels)
{
    if (image.empty())
    {
        return;
    }

    if (!results.detections.empty() && !results.label_mask.empty())
    {
        drawLabelMaskOverlay(image, results.label_mask);
    }

    drawDetectionBoxes(image, results.detections, labels);
}

void drawPoseResults(cv::Mat &image, const PoseResult &results, const data::ClassLabelTable *labels)
{
    if (image.empty())
    {
        return;
    }

    static const int skeleton[38] = {16, 14, 14, 12, 17, 15, 15, 13, 12, 13, 6, 12, 7, 13, 6, 7, 6, 8,
                                     7, 9, 8, 10, 9, 11, 2, 3, 1, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7};

    const cv::Scalar color_box(255, 0, 0);
    const cv::Scalar color_line(0, 165, 255);
    const cv::Scalar color_kp(0, 255, 255);
    const float kpt_conf_thr = 0.25f;

    for (const PoseDetection &det : results.detections)
    {
        cv::rectangle(image,
                      cv::Point(det.box.left, det.box.top),
                      cv::Point(det.box.right, det.box.bottom),
                      color_box,
                      2);

        const std::string label = formatPoseCaption(det, labels);
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
            if (det.keypoints[a].score >= kpt_conf_thr && det.keypoints[b].score >= kpt_conf_thr)
            {
                cv::line(image,
                         cv::Point(static_cast<int>(det.keypoints[a].x),
                                   static_cast<int>(det.keypoints[a].y)),
                         cv::Point(static_cast<int>(det.keypoints[b].x),
                                   static_cast<int>(det.keypoints[b].y)),
                         color_line,
                         2);
            }
        }
        for (int j = 0; j < 17; ++j)
        {
            if (det.keypoints[j].score >= kpt_conf_thr)
            {
                cv::circle(image,
                           cv::Point(static_cast<int>(det.keypoints[j].x),
                                     static_cast<int>(det.keypoints[j].y)),
                           3,
                           color_kp,
                           -1);
            }
        }
    }
}

} // namespace utils
} // namespace deploy_percept
