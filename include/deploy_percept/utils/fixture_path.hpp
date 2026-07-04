#pragma once

#include <filesystem>
#include <string>

namespace deploy_percept
{
    namespace utils
    {

        /**
         * Resolve a path under tests/fixtures (source tree) or share/percept/fixtures (install).
         *
         * @p relative e.g. "unit/post_process/yolov5_seg/model_outputs.npz"
         *              or "integration/awnn/yolov5_detect/yolov5.nb"
         */
        std::filesystem::path fixture_path(const std::string &relative);

    } // namespace utils
} // namespace deploy_percept
