#include "deploy_percept/utils/fixture_path.hpp"

#include "deploy_percept/utils/install_paths.hpp"

namespace fs = std::filesystem;

namespace deploy_percept
{
    namespace utils
    {

        namespace
        {

            bool fixture_file_exists(const fs::path &path)
            {
                std::error_code ec;
                return fs::is_regular_file(path, ec);
            }

        } // namespace

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

    } // namespace utils
} // namespace deploy_percept
