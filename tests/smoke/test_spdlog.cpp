#include <gtest/gtest.h>

#include <spdlog/spdlog.h>

TEST(Smoke, SpdlogLink) {
    spdlog::set_pattern("[%l] %v");
    spdlog::info("spdlog version {}", SPDLOG_VERSION);
    SUCCEED();
}
