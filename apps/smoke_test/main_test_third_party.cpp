#include <cstdio>

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

int main() {
    std::printf("deploy_percept smoke test (third party)\n");

    spdlog::set_pattern("[%l] %v");
    spdlog::info("spdlog version {}", SPDLOG_VERSION);

    constexpr const char* kYaml = "smoke_test:\n  status: ok\n";
    const YAML::Node root = YAML::Load(kYaml);
    if (!root["smoke_test"] || root["smoke_test"]["status"].as<std::string>() != "ok") {
        std::fprintf(stderr, "yaml-cpp parse failed\n");
        return 1;
    }

    std::printf("  spdlog   : OK\n");
    std::printf("  yaml-cpp : OK\n");
    std::printf("OK\n");
    return 0;
}
