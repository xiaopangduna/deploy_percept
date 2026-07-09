# FindThirdParty.cmake - 第三方库查找入口（仅负责 Find，不做编译门禁）
#
# THIRD_PARTY_* 路径由各 toolchain 设置，此处不重复推导。
# 编译门禁由根 CMakeLists.txt 与各子目录 CMakeLists.txt 直接用 option + *_FOUND 判断。

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
#     include(FindRgaCustom)
# endif()

# 与平台无关：第三方 Find（非致命）
# cnpy 依赖 ZLIB::ZLIB，须先 include FindZlibCustom
include(FindZlibCustom)
include(FindCnpyCustom)
include(FindOpenCVCustom)
include(FindSpdlogCustom)
include(FindGTestCustom)
include(FindYamlCppCustom)

message(STATUS "已完成第三方库配置")
