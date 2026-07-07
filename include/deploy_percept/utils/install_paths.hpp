#pragma once

#include <filesystem>

namespace deploy_percept
{
    namespace utils
    {

        /** Install prefix inferred from /proc/self/exe (bin/, tests/, benchmarks/). */
        std::filesystem::path install_prefix_from_executable();

        /** Data root: share/percept (from PERCEPT_ROOT or install layout). */
        std::filesystem::path share_percept_root();

        /** Output dir for install tree: var/percept/output (or PERCEPT_OUTPUT_DIR). */
        std::filesystem::path install_output_dir();

    } // namespace utils
} // namespace deploy_percept
