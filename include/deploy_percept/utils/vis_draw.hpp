#pragma once

#include "deploy_percept/common/types.hpp"
#include "deploy_percept/data/class_labels.hpp"

#include <opencv2/core/mat.hpp>

namespace deploy_percept
{
namespace utils
{

void drawDetectionResults(cv::Mat &image,
                          const DetectionResult &results,
                          const data::ClassLabelTable *labels = nullptr);

void drawSegmentationResults(cv::Mat &image,
                             const SegmentationResult &results,
                             const data::ClassLabelTable *labels = nullptr);

void drawPoseResults(cv::Mat &image,
                     const PoseResult &results,
                     const data::ClassLabelTable *labels = nullptr);

} // namespace utils
} // namespace deploy_percept
