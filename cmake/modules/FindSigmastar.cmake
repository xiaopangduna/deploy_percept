# FindSigmastar.cmake - Sigmastar MI / IPU SDK（CoMake Pi D1 / SSD2355 / pcupid）
#
# 准备：将 Linux SDK 的 project/release 置于（或软链到）
#   ${THIRD_PARTY_PLATFORM_DIR}/sigmastar/release
#
# 例如：
#   mkdir -p third_party/armv7l-linux-gnueabihf_comake_pi_d1_ssd2355/sigmastar
#   ln -sfn /path/to/SourceCode/project/release \
#     third_party/armv7l-linux-gnueabihf_comake_pi_d1_ssd2355/sigmastar/release
#
# 提供：SIGMASTAR_FOUND、sigmastar::mi

set(SIGMASTAR_FOUND FALSE)

if(NOT THIRD_PARTY_PLATFORM_TAG STREQUAL "armv7l-linux-gnueabihf_comake_pi_d1_ssd2355")
    message(STATUS
        "FindSigmastar: skip platform '${THIRD_PARTY_PLATFORM_TAG}' "
        "(only armv7l-linux-gnueabihf_comake_pi_d1_ssd2355)")
    return()
endif()

set(_SIGMASTAR_SDK_ROOT "${THIRD_PARTY_PLATFORM_DIR}/sigmastar/release")
set(_SIGMASTAR_MI_LIB_DIR
    "${_SIGMASTAR_SDK_ROOT}/chip/pcupid/dispcam/common/glibc/12.4.0/arm/release/mi_libs/static")
set(_SIGMASTAR_COMMON_LIB_DIR
    "${_SIGMASTAR_SDK_ROOT}/chip/pcupid/sigma_common_libs/glibc/12.4.0/arm/release/static")
set(_SIGMASTAR_INCLUDE_DIR "${_SIGMASTAR_SDK_ROOT}/include")
set(_SIGMASTAR_IPU_INCLUDE_DIR "${_SIGMASTAR_INCLUDE_DIR}/ipu/pcupid")

if(NOT EXISTS "${_SIGMASTAR_IPU_INCLUDE_DIR}/mi_ipu.h"
   OR NOT EXISTS "${_SIGMASTAR_MI_LIB_DIR}/libmi_ipu.a"
   OR NOT EXISTS "${_SIGMASTAR_COMMON_LIB_DIR}/libcam_os_wrapper.a")
    message(STATUS "==============================================================================")
    message(STATUS "Sigmastar MI/IPU not found")
    message(STATUS "  expected root    : ${_SIGMASTAR_SDK_ROOT}")
    message(STATUS "  expected include : ${_SIGMASTAR_IPU_INCLUDE_DIR}/mi_ipu.h")
    message(STATUS "  expected mi libs : ${_SIGMASTAR_MI_LIB_DIR}/libmi_ipu.a")
    message(STATUS "  expected common  : ${_SIGMASTAR_COMMON_LIB_DIR}/libcam_os_wrapper.a")
    message(STATUS "  layout           : Linux SDK project/release under sigmastar/")
    message(STATUS "==============================================================================")
    unset(_SIGMASTAR_SDK_ROOT)
    unset(_SIGMASTAR_MI_LIB_DIR)
    unset(_SIGMASTAR_COMMON_LIB_DIR)
    unset(_SIGMASTAR_INCLUDE_DIR)
    unset(_SIGMASTAR_IPU_INCLUDE_DIR)
    return()
endif()

if(TARGET sigmastar::mi)
    set(SIGMASTAR_FOUND TRUE)
    return()
endif()

add_library(sigmastar_mi INTERFACE)
add_library(sigmastar::mi ALIAS sigmastar_mi)

target_compile_definitions(sigmastar_mi INTERFACE
    SGS_CHIP_PCUPID
    INTERFACE_IPU
    INTERFACE_SYS
)

target_include_directories(sigmastar_mi INTERFACE
    "${_SIGMASTAR_INCLUDE_DIR}"
    "${_SIGMASTAR_IPU_INCLUDE_DIR}"
)

target_link_directories(sigmastar_mi INTERFACE
    "${_SIGMASTAR_MI_LIB_DIR}"
    "${_SIGMASTAR_COMMON_LIB_DIR}"
)

# OpenDLA detection demo 所需最小 MI 依赖（与 opendla build.mk / yolov8.mk 对齐）
target_link_libraries(sigmastar_mi INTERFACE
    -Wl,--start-group
    mi_ipu
    mi_sys
    mi_common
    cam_fs_wrapper
    cam_os_wrapper
    -Wl,--end-group
    pthread
    m
    dl
    rt
)

set(SIGMASTAR_FOUND TRUE)
set(SIGMASTAR_SDK_ROOT "${_SIGMASTAR_SDK_ROOT}" CACHE INTERNAL "Sigmastar SDK release root")
set(SIGMASTAR_MI_LIB_DIR "${_SIGMASTAR_MI_LIB_DIR}" CACHE INTERNAL "Sigmastar MI static lib dir")

message(STATUS "==============================================================================")
message(STATUS "Sigmastar MI/IPU found: ${THIRD_PARTY_PLATFORM_TAG}")
message(STATUS "  sdk root : ${_SIGMASTAR_SDK_ROOT}")
message(STATUS "  mi libs  : ${_SIGMASTAR_MI_LIB_DIR}")
message(STATUS "  common   : ${_SIGMASTAR_COMMON_LIB_DIR}")
message(STATUS "  include  : ${_SIGMASTAR_INCLUDE_DIR}")
message(STATUS "  ipu      : ${_SIGMASTAR_IPU_INCLUDE_DIR}")
message(STATUS "  target   : sigmastar::mi")
message(STATUS "==============================================================================")

unset(_SIGMASTAR_SDK_ROOT)
unset(_SIGMASTAR_MI_LIB_DIR)
unset(_SIGMASTAR_COMMON_LIB_DIR)
unset(_SIGMASTAR_INCLUDE_DIR)
unset(_SIGMASTAR_IPU_INCLUDE_DIR)
