# FindAwnn.cmake - Allwinner VIPLite（NPU 运行时）
#
# 准备：将 awnpu_model_zoo 中的 common/npuruntime 整包置于
#   ${THIRD_PARTY_PLATFORM_DIR}/awnn/npuruntime
#
# 下载来源（由维护者填写，Find 模块不自动下载）：
# https://open.allwinnertech.com/#/otherDir/aiToolList

#
# 提供：AWNN_FOUND、AWNN::VIPLite（头文件 + libNBGlinker.so + libVIPhal.so）
# install 打包上述 .so 至 ${CMAKE_INSTALL_LIBDIR}

include(GNUInstallDirs)

set(AWNN_FOUND FALSE)

set(AWNN_NPU_RUNTIME_ROOT "${THIRD_PARTY_PLATFORM_DIR}/awnn/npuruntime")

# AWNN_PLATFORM（toolchain 设置）→ npuruntime 内 SOC / 架构目录
if(AWNN_PLATFORM STREQUAL "a733")
    set(_AWNN_SOC "A733")
    set(_AWNN_LIB_REL "lib_linux_aarch64")
    set(AWNN_VIPLITE_INCLUDE_DIR "${AWNN_NPU_RUNTIME_ROOT}/include")
else()
    message(STATUS "AWNN: 不支持的 AWNN_PLATFORM '${AWNN_PLATFORM}'")
    return()
endif()

set(AWNN_VIPLITE_LIB_DIR "${AWNN_NPU_RUNTIME_ROOT}/${_AWNN_LIB_REL}/${_AWNN_SOC}")
set(AWNN_VIPLITE_LIBS
    "${AWNN_VIPLITE_LIB_DIR}/libNBGlinker.so"
    "${AWNN_VIPLITE_LIB_DIR}/libVIPhal.so")

if(NOT EXISTS "${AWNN_VIPLITE_INCLUDE_DIR}/vip_lite.h"
   OR NOT EXISTS "${AWNN_VIPLITE_LIB_DIR}/libNBGlinker.so"
   OR NOT EXISTS "${AWNN_VIPLITE_LIB_DIR}/libVIPhal.so")
    message(STATUS "==============================================================================")
    message(STATUS "AWNN VIPLite not found")
    message(STATUS "  AWNN_PLATFORM       : ${AWNN_PLATFORM}")
    message(STATUS "  expected root       : ${AWNN_NPU_RUNTIME_ROOT}")
    message(STATUS "  expected include    : ${AWNN_VIPLITE_INCLUDE_DIR}")
    message(STATUS "  expected lib dir    : ${AWNN_VIPLITE_LIB_DIR}")
    message(STATUS "  layout              : awnpu_model_zoo/common/npuruntime (full tree)")
    if(DEFINED AWNN_NPU_RUNTIME_PACKAGE_URL)
        message(STATUS "  download            : ${AWNN_NPU_RUNTIME_PACKAGE_URL}")
    else()
        message(STATUS "  download            : (set AWNN_NPU_RUNTIME_PACKAGE_URL in FindAwnn.cmake)")
    endif()
    message(STATUS "==============================================================================")
    return()
endif()

if(TARGET AWNN::VIPLite)
    set(AWNN_FOUND TRUE)
    return()
endif()

add_library(AWNN_VIPLite INTERFACE)
add_library(AWNN::VIPLite ALIAS AWNN_VIPLite)

target_include_directories(AWNN_VIPLite INTERFACE "${AWNN_VIPLITE_INCLUDE_DIR}")
target_link_libraries(AWNN_VIPLite INTERFACE ${AWNN_VIPLITE_LIBS})

install(FILES ${AWNN_VIPLITE_LIBS} DESTINATION ${CMAKE_INSTALL_LIBDIR})

set(AWNN_FOUND TRUE)

message(STATUS "==============================================================================")
message(STATUS "AWNN VIPLite found successfully")
message(STATUS "  platform: ${AWNN_PLATFORM}")
message(STATUS "  soc     : ${_AWNN_SOC}")
message(STATUS "  root    : ${AWNN_NPU_RUNTIME_ROOT}")
message(STATUS "  lib dir : ${AWNN_VIPLITE_LIB_DIR}")
message(STATUS "  include : ${AWNN_VIPLITE_INCLUDE_DIR}")
message(STATUS "  libs    : libNBGlinker.so libVIPhal.so")
message(STATUS "  target  : AWNN::VIPLite")
message(STATUS "==============================================================================")
