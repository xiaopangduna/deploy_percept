#pragma once

#include <filesystem>
#include <string>

namespace percept {
namespace test {

/** Install prefix inferred from /proc/self/exe (tests/, benchmarks/). */
std::filesystem::path install_prefix_from_executable();

/** Data root: share/percept (from PERCEPT_ROOT or install layout). */
std::filesystem::path share_percept_root();

/** Output dir for install tree: var/percept/output (or PERCEPT_OUTPUT_DIR). */
std::filesystem::path install_output_dir();

/**
 * Resolve a file under tests/fixtures (source tree) or share/percept/fixtures (install).
 *
 * @p relative e.g. "unit/post_process/yolov5_seg/model_outputs.npz"
 *              or "integration/awnn/yolov5_a733.nb"
 */
std::filesystem::path fixture_path(const std::string &relative);

/**
 * Resolve a directory under the same fixture roots as fixture_path().
 *
 * @p relative e.g. "integration/sigmastar"
 */
std::filesystem::path fixture_dir(const std::string &relative);

inline std::filesystem::path fixture_data(const std::string &relative)
{
    return fixture_path(relative);
}

/** Test output directory (ctest: tmp/; install: var/percept/output). */
std::filesystem::path output_dir();

} // namespace test
} // namespace percept
