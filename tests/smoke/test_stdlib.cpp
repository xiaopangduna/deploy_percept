#include <gtest/gtest.h>

#include <unistd.h>

TEST(Smoke, StdlibRuntime)
{
    EXPECT_GT(getpid(), 0);
}
