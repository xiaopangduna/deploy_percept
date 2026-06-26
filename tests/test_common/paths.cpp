#include "paths.hpp"

#include <cstdlib>
#include <unistd.h>

namespace fs = std::filesystem;

namespace percept {
namespace test {

namespace {

fs::path root_from_env()
{
    if (const char *env = std::getenv("PERCEPT_ROOT"))
    {
        return env;
    }
    return {};
}

fs::path install_prefix_from_executable()
{
    char buf[4096];
    const ssize_t len = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len <= 0)
    {
        return {};
    }
    buf[len] = '\0';

    const fs::path exe_dir = fs::path(buf).parent_path();

    // <prefix>/share/percept/tests/<exe>
    if (exe_dir.filename() == "tests")
    {
        const fs::path percept_dir = exe_dir.parent_path();
        if (percept_dir.filename() == "percept")
        {
            return percept_dir.parent_path().parent_path();
        }
    }

    // <prefix>/bin/<exe>（兼容旧 layout）
    if (exe_dir.filename() == "bin")
    {
        return exe_dir.parent_path();
    }

    return {};
}

} // namespace

fs::path resolve_root()
{
    if (const fs::path env_root = root_from_env(); !env_root.empty())
    {
        return env_root;
    }

#ifdef PERCEPT_ROOT
    return PERCEPT_ROOT;
#else
    const fs::path prefix = install_prefix_from_executable();
    if (!prefix.empty())
    {
        return prefix / "share" / "percept";
    }
    return {};
#endif
}

fs::path app_data(const std::string &relative)
{
    return resolve_root() / "apps" / relative;
}

fs::path output_dir()
{
    if (const char *env = std::getenv("PERCEPT_OUTPUT_DIR"))
    {
        return env;
    }

#ifdef PERCEPT_ROOT
    return fs::path(PERCEPT_ROOT) / "tmp";
#else
    const fs::path prefix = install_prefix_from_executable();
    if (!prefix.empty())
    {
        return prefix / "var" / "percept" / "output";
    }
    return "tmp";
#endif
}

} // namespace test
} // namespace percept
