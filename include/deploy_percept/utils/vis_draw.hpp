#pragma once

#include "deploy_percept/common/types.hpp"

#include <opencv2/core/mat.hpp>

namespace deploy_percept
{
namespace utils
{

void drawDetectionResults(cv::Mat &image, const DetectionResult &results);
void drawDetectionResults(cv::Mat &image, const SegmentationResult &results);
void drawPoseResults(cv::Mat &image, const PoseResult &results);

} // namespace utils
} // namespace deploy_percept
