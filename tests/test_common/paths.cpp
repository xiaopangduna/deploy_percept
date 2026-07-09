#include "paths.hpp"

#include <cstdlib>
#include <unistd.h>

namespace fs = std::filesystem;

namespace percept {
namespace test {

namespace {

fs::path percept_root_from_env()
{
    if (const char *env = std::getenv("PERCEPT_ROOT"))
    {
        return env;
    }
    return {};
}

bool fixture_file_exists(const fs::path &path)
{
    std::error_code ec;
    return fs::is_regular_file(path, ec);
}

} // namespace

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

    if (exe_dir.filename() == "benchmarks")
    {
        const fs::path parent = exe_dir.parent_path();
        if (parent.filename() == "tests")
        {
            return parent.parent_path();
        }
        return parent;
    }

    if (exe_dir.filename() == "tests")
    {
        const fs::path parent = exe_dir.parent_path();
        if (parent.filename() == "percept")
        {
            return parent.parent_path().parent_path();
        }
        return parent;
    }

    if (exe_dir.filename() == "bin")
    {
        return exe_dir.parent_path();
    }

    return {};
}

fs::path share_percept_root()
{
    if (const fs::path env_root = percept_root_from_env(); !env_root.empty())
    {
        if (env_root.filename() == "percept")
        {
            return env_root;
        }
        const fs::path as_share = env_root / "share" / "percept";
        if (fs::is_directory(as_share))
        {
            return as_share;
        }
        return env_root;
    }

    const fs::path prefix = install_prefix_from_executable();
    if (!prefix.empty())
    {
        return prefix / "share" / "percept";
    }

    return {};
}

fs::path install_output_dir()
{
    if (const char *env = std::getenv("PERCEPT_OUTPUT_DIR"))
    {
        return env;
    }

    const fs::path prefix = install_prefix_from_executable();
    if (!prefix.empty())
    {
        return prefix / "var" / "percept" / "output";
    }

    return {};
}

fs::path fixture_path(const std::string &relative)
{
    const fs::path root = share_percept_root();

    if (!root.empty())
    {
        const fs::path installed = root / "fixtures" / relative;
        if (fixture_file_exists(installed))
        {
            return installed;
        }

        const fs::path source_tree = root / "tests" / "fixtures" / relative;
        if (fixture_file_exists(source_tree))
        {
            return source_tree;
        }

        return installed;
    }

    return fs::path("tests/fixtures") / relative;
}

fs::path output_dir()
{
#ifdef PERCEPT_ROOT
    if (const char *env = std::getenv("PERCEPT_OUTPUT_DIR"))
    {
        return env;
    }
    return fs::path(PERCEPT_ROOT) / "tmp";
#else
    if (const fs::path dir = install_output_dir(); !dir.empty())
    {
        return dir;
    }
    return "tmp";
#endif
}

} // namespace test
} // namespace percept
