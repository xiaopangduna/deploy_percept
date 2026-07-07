#include "paths.hpp"

#include "deploy_percept/utils/install_paths.hpp"

namespace fs = std::filesystem;

namespace percept {
namespace test {

fs::path output_dir()
{
#ifdef PERCEPT_ROOT
    if (const char *env = std::getenv("PERCEPT_OUTPUT_DIR"))
    {
        return env;
    }
    return fs::path(PERCEPT_ROOT) / "tmp";
#else
    if (const fs::path dir = deploy_percept::utils::install_output_dir(); !dir.empty())
    {
        return dir;
    }
    return "tmp";
#endif
}

} // namespace test
} // namespace percept
