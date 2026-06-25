# FindAwnn.cmake - Allwinner VIPLite（NPU 运行时）
#
# 准备：将 viplite-tina 拷贝至
#   third_party/<platform>/awnn/viplite-tina
#
# 提供：AWNN_FOUND、AWNN::VIPLite（头文件 + libNBGlinker.so + libVIPhal.so）
# install 打包上述 .so 至 ${CMAKE_INSTALL_LIBDIR}

include(GNUInstallDirs)

set(AWNN_FOUND FALSE)

if(AWNN_PLATFORM STREQUAL "a733")
    set(AWNN_VIPLITE_LIB_DIR
        "${THIRD_PARTY_PLATFORM_DIR}/awnn/viplite-tina/lib/aarch64-none-linux-gnu/v2.0")
else()
    message(STATUS "AWNN: 不支持的 AWNN_PLATFORM '${AWNN_PLATFORM}'")
    return()
endif()

set(AWNN_VIPLITE_INCLUDE_DIR "${AWNN_VIPLITE_LIB_DIR}/inc")
set(AWNN_VIPLITE_LIBS
    "${AWNN_VIPLITE_LIB_DIR}/libNBGlinker.so"
    "${AWNN_VIPLITE_LIB_DIR}/libVIPhal.so")

if(NOT EXISTS "${AWNN_VIPLITE_INCLUDE_DIR}/vip_lite.h"
   OR NOT EXISTS "${AWNN_VIPLITE_LIB_DIR}/libNBGlinker.so"
   OR NOT EXISTS "${AWNN_VIPLITE_LIB_DIR}/libVIPhal.so")
    message(STATUS "==============================================================================")
    message(STATUS "AWNN VIPLite not found: ${AWNN_VIPLITE_LIB_DIR}")
    message(STATUS "  platform: ${AWNN_PLATFORM}")
    message(STATUS "  Prepare viplite-tina under third_party/<platform>/awnn/")
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
message(STATUS "  lib dir : ${AWNN_VIPLITE_LIB_DIR}")
message(STATUS "  include : ${AWNN_VIPLITE_INCLUDE_DIR}")
message(STATUS "  libs    : libNBGlinker.so libVIPhal.so")
message(STATUS "  target  : AWNN::VIPLite")
message(STATUS "==============================================================================")
