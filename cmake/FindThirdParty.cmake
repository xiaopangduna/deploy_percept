# FindThirdParty.cmake - 第三方库查找入口
#
# THIRD_PARTY_* 路径由各 toolchain 设置，此处不重复推导。

if(NOT DEFINED THIRD_PARTY_PLATFORM_DIR)
    message(FATAL_ERROR
        "THIRD_PARTY_PLATFORM_DIR is not set. "
        "Please configure a toolchain file (cmake/*-toolchain.cmake).")
endif()

message(STATUS "==============================================================================")
message(STATUS "第三方库配置信息:")
message(STATUS "  - THIRD_PARTY_PLATFORM_DIR: ${THIRD_PARTY_PLATFORM_DIR}")

if(THIRD_PARTY_PLATFORM_TAG STREQUAL "armv7l-SSC375")
    include(FindSigmastar)
endif()

# include(FindSpdlogCustom)
# include(FindYamlCppCustom)

# include(FindOpenCVCustom)

# # 导入自定义的查找模块
# include(FindCnpyCustom)
# include(FindZlibCustom)
# include(FindRknn)
# include(FindRgaCustom)
# include(FindGTestCustom)
# include(FindNlohmannJson)

message(STATUS "已完成第三方库配置")
