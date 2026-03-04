#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

namespace deploy_percept
{
    namespace post_process
    {

        struct BoxRect
        {
            int left = 0;
            int top = 0;
            int right = 0;
            int bottom = 0;
        };

        struct DetectionObject
        {
            float prop = 0.0f;
            int cls_id = 0; 
            char name[16] = {}; 
            BoxRect box{};
        };

        struct ResultGroup
        {
            int id = 0;
            int count = 0;
            std::vector<DetectionObject> detection_objects;    
            std::vector<uint8_t> segmentation_mask; 
        };

    } // namespace post_process
} // namespace deploy_percept