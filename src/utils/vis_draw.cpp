#include "deploy_percept/utils/vis_draw.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <string>

namespace deploy_percept
{
namespace utils
{

void drawDetectionResults(cv::Mat &image, const ResultGroup &results)
{
    if (image.empty())
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

    if (results.count >= 1 && !results.segmentation_mask.empty())
    {
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                const int mask_value = results.segmentation_mask[h * width + w];

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

    for (int i = 0; i < results.count; i++)
    {
        const DetectionObject *det_result = &results.detection_objects[i];

        const cv::Scalar color = cv::Scalar(class_colors[det_result->cls_id % 20][2],
                                            class_colors[det_result->cls_id % 20][1],
                                            class_colors[det_result->cls_id % 20][0]);

        cv::rectangle(image,
                      cv::Point(det_result->box.left, det_result->box.top),
                      cv::Point(det_result->box.right, det_result->box.bottom),
                      color,
                      2);

        const std::string label = "Class " + std::to_string(det_result->cls_id) + " " +
                                  std::to_string(det_result->prop * 100) + "%";
        int baseline = 0;
        const cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(image,
                      cv::Point(det_result->box.left, det_result->box.top - textSize.height - 10),
                      cv::Point(det_result->box.left + textSize.width, det_result->box.top),
                      color,
                      -1);
        cv::putText(image,
                    label,
                    cv::Point(det_result->box.left, det_result->box.top - 5),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(255, 255, 255),
                    1);
    }
}

void drawPoseResults(cv::Mat &image, const PoseResultGroup &results)
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

    for (size_t i = 0; i < results.objects.size(); ++i)
    {
        const PoseDetectionObject &det = results.objects[i];
        cv::rectangle(image,
                      cv::Point(det.box.left, det.box.top),
                      cv::Point(det.box.right, det.box.bottom),
                      color_box,
                      2);

        const std::string cls_name = (det.cls_id == 0) ? "人" : ("类" + std::to_string(det.cls_id));
        const float prop_pct = det.prop * 100.f;
        const std::string label = cls_name + " id=" + std::to_string(det.cls_id) + " " + std::to_string(prop_pct) + "%";
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

} // namespace utils
} // namespace deploy_percept
