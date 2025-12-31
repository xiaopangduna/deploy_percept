#include "deploy_percept/pre_process/image_pre_process_rga.hpp"
#include <algorithm>


namespace deploy_percept
{
    namespace pre_process
    {

        int resize_rga(rga_buffer_t &src, rga_buffer_t &dst, const cv::Mat &image, cv::Mat &resized_image, const cv::Size &target_size)
        {
            im_rect src_rect;
            im_rect dst_rect;
            memset(&src_rect, 0, sizeof(src_rect));
            memset(&dst_rect, 0, sizeof(dst_rect));
            size_t img_width = image.cols;
            size_t img_height = image.rows;
            if (image.type() != CV_8UC3)
            {
                printf("source image type is %d!\n", image.type());
                return -1;
            }
            size_t target_width = target_size.width;
            size_t target_height = target_size.height;
            src = wrapbuffer_virtualaddr((void *)image.data, img_width, img_height, RK_FORMAT_RGB_888);
            dst = wrapbuffer_virtualaddr((void *)resized_image.data, target_width, target_height, RK_FORMAT_RGB_888);
            int ret = imcheck(src, dst, src_rect, dst_rect);
            if (IM_STATUS_NOERROR != ret)
            {
                fprintf(stderr, "rga check error! %s", imStrError((IM_STATUS)ret));
                return -1;
            }
            IM_STATUS STATUS = imresize(src, dst);
            return 0;
        }

    } // namespace pre_process
} // namespace deploy_percept