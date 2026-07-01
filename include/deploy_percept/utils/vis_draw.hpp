#pragma once

#include "deploy_percept/types.hpp"

#include <opencv2/core/mat.hpp>

namespace deploy_percept
{
namespace utils
{

void drawDetectionResults(cv::Mat &image, const ResultGroup &results);
void drawPoseResults(cv::Mat &image, const PoseResultGroup &results);

} // namespace utils
} // namespace deploy_percept
