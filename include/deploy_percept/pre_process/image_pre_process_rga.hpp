#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include "im2d.h"
#include "rga.h"

namespace deploy_percept
{
    namespace pre_process
    {
        int resize_rga(rga_buffer_t &src, rga_buffer_t &dst, const cv::Mat &image, cv::Mat &resized_image, const cv::Size &target_size);

    } // namespace pre_process
} // namespace deploy_percept