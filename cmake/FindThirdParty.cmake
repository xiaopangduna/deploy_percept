# FindThirdParty.cmake - 第三方库查找入口
#
# THIRD_PARTY_* 路径由各 toolchain 设置，此处不重复推导。

if(NOT DEFINED THIRD_PARTY_PLATFORM_DIR)
    message(FATAL_ERROR
        "THIRD_PARTY_PLATFORM_DIR is not set. "
        "Please configure a toolchain file (cmake/toolchains/*-toolchain.cmake).")
endif()

message(STATUS "==============================================================================")
message(STATUS "第三方库配置信息:")
message(STATUS "  - THIRD_PARTY_PLATFORM_DIR: ${THIRD_PARTY_PLATFORM_DIR}")

# 平台专属 NPU 运行时（一平台最多一个；未 include 则该平台无对应引擎）
# TODO: Sigmastar SSC375
# if(THIRD_PARTY_PLATFORM_TAG STREQUAL "armv7l-SSC375")
#     include(FindSigmastar)
# endif()

if(THIRD_PARTY_PLATFORM_TAG STREQUAL "aarch64-linux-gnu_orange_pi_4_pro_a733")
    include(FindAwnn)
endif()

# TODO: RKNN（瑞芯微）
# if(THIRD_PARTY_PLATFORM_TAG STREQUAL "aarch64-linux-gnu_rk3588")
#     include(FindRknn)
#     include(FindRgaCustom)
# endif()

# 与平台无关：cnpy / OpenCV / spdlog 统一 Find（非致命）
include(FindCnpyCustom)
set(SPDLOG_FIND_REQUIRED OFF)
include(FindSpdlogCustom)
include(FindOpenCVCustom)

set(PERCEPT_UTILS_READY FALSE)
if(CNPY_FOUND AND OpenCV_FOUND AND spdlog_FOUND)
    set(PERCEPT_UTILS_READY TRUE)
endif()
message(STATUS "PERCEPT_UTILS_READY: ${PERCEPT_UTILS_READY}"
    " (cnpy=${CNPY_FOUND} opencv=${OpenCV_FOUND} spdlog=${spdlog_FOUND})")

# 测试分层门禁（GTest 仅 ENABLE_TESTS 时 Find）
set(PERCEPT_SMOKE_TESTS_READY FALSE)
set(PERCEPT_UNIT_TESTS_READY FALSE)
set(PERCEPT_INTEGRATION_TESTS_READY FALSE)

if(ENABLE_TESTS)
    include(FindGTestCustom)

    if(GTest_FOUND)
        set(PERCEPT_SMOKE_TESTS_READY TRUE)
    endif()

    if(GTest_FOUND AND PERCEPT_UTILS_READY)
        set(PERCEPT_UNIT_TESTS_READY TRUE)
    endif()

    if(GTest_FOUND AND AWNN_FOUND AND PERCEPT_UTILS_READY)
        set(PERCEPT_INTEGRATION_TESTS_READY TRUE)
    endif()

    message(STATUS "tests readiness:"
        " smoke=${PERCEPT_SMOKE_TESTS_READY}"
        " unit=${PERCEPT_UNIT_TESTS_READY}"
        " integration=${PERCEPT_INTEGRATION_TESTS_READY}")
endif()

set(PERCEPT_BENCHMARKS_READY FALSE)
if(AWNN_FOUND AND PERCEPT_UTILS_READY)
    set(PERCEPT_BENCHMARKS_READY TRUE)
endif()

# TODO: yaml-cpp（RKNN app 启用后再 include(FindYamlCppCustom)）

message(STATUS "已完成第三方库配置")
