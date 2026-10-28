# FindSigmastar.cmake - Sigmastar MI / IPU SDK
#
# 概念分层：
#   THIRD_PARTY_PLATFORM_TAG（板级）→ SIGMASTAR_CHIP_ARCH（SGS 芯片架构）
#   → 库相对路径 / include/ipu/<arch>/ / SGS_CHIP_<ARCH>
#
# 当前映射：
#   armv7l-linux-gnueabihf_comake_pi_d1_ssd2355 → pcupid → SGS_CHIP_PCUPID
#   armv7l-linux-gnueabihf_ssc308qe             → iford  → SGS_CHIP_IFORD
#
# SDK 唯一入口（软链或拷贝均可）：
#   ${THIRD_PARTY_PLATFORM_DIR}/sigmastar/release
#
#   mkdir -p third_party/<platform>/sigmastar
#   ln -sfn /path/to/SourceCode/project/release \
#     third_party/<platform>/sigmastar/release
#
# 提供：
#   SIGMASTAR_FOUND、SIGMASTAR_CHIP_ARCH、sigmastar::mi
#   单摄预览另外提供 sigmastar::sensor_pipeline
#   （sensor / vif / isp / scl / disp + 3A；不含 vdisp、ipu 画框、gfx）

set(SIGMASTAR_FOUND FALSE)
set(SIGMASTAR_CHIP_ARCH "")

# ---------------------------------------------------------------------------
# 1) 平台 → SGS 芯片架构
# ---------------------------------------------------------------------------
if(THIRD_PARTY_PLATFORM_TAG STREQUAL "armv7l-linux-gnueabihf_comake_pi_d1_ssd2355")
    set(SIGMASTAR_CHIP_ARCH "pcupid")
elseif(THIRD_PARTY_PLATFORM_TAG STREQUAL "armv7l-linux-gnueabihf_ssc308qe")
    set(SIGMASTAR_CHIP_ARCH "iford")
else()
    message(STATUS
        "FindSigmastar: skip platform '${THIRD_PARTY_PLATFORM_TAG}' "
        "(no SGS chip arch mapping)")
    return()
endif()

# ---------------------------------------------------------------------------
# 2) 架构 → 相对路径 + 编译宏
# ---------------------------------------------------------------------------
set(_SIGMASTAR_MI_LIB_REL "")
set(_SIGMASTAR_COMMON_LIB_REL "")
set(_SIGMASTAR_CHIP_DEFINE "")

if(SIGMASTAR_CHIP_ARCH STREQUAL "pcupid")
    set(_SIGMASTAR_MI_LIB_REL
        "chip/pcupid/dispcam/common/glibc/12.4.0/arm/release/mi_libs/static")
    set(_SIGMASTAR_COMMON_LIB_REL
        "chip/pcupid/sigma_common_libs/glibc/12.4.0/arm/release/static")
    set(_SIGMASTAR_CHIP_DEFINE "SGS_CHIP_PCUPID")
elseif(SIGMASTAR_CHIP_ARCH STREQUAL "iford")
    set(_SIGMASTAR_MI_LIB_REL
        "chip/iford/ipc/common/glibc/11.1.0/release/mi_libs/static")
    set(_SIGMASTAR_COMMON_LIB_REL
        "chip/iford/sigma_common_libs/glibc/11.1.0/release/static")
    set(_SIGMASTAR_CHIP_DEFINE "SGS_CHIP_IFORD")
else()
    message(STATUS
        "FindSigmastar: unknown SGS chip arch '${SIGMASTAR_CHIP_ARCH}'")
    set(SIGMASTAR_CHIP_ARCH "")
    return()
endif()

# ---------------------------------------------------------------------------
# 3) 解析 SDK（只认 sigmastar/release）
# ---------------------------------------------------------------------------
set(_SIGMASTAR_SDK_ROOT "${THIRD_PARTY_PLATFORM_DIR}/sigmastar/release")
set(_SIGMASTAR_MI_LIB_DIR "${_SIGMASTAR_SDK_ROOT}/${_SIGMASTAR_MI_LIB_REL}")
set(_SIGMASTAR_COMMON_LIB_DIR "${_SIGMASTAR_SDK_ROOT}/${_SIGMASTAR_COMMON_LIB_REL}")
set(_SIGMASTAR_INCLUDE_DIR "${_SIGMASTAR_SDK_ROOT}/include")
set(_SIGMASTAR_IPU_INCLUDE_DIR
    "${_SIGMASTAR_INCLUDE_DIR}/ipu/${SIGMASTAR_CHIP_ARCH}")

if(NOT EXISTS "${_SIGMASTAR_IPU_INCLUDE_DIR}/mi_ipu.h"
   OR NOT EXISTS "${_SIGMASTAR_MI_LIB_DIR}/libmi_ipu.a"
   OR NOT EXISTS "${_SIGMASTAR_COMMON_LIB_DIR}/libcam_os_wrapper.a")
    message(STATUS "==============================================================================")
    message(STATUS "Sigmastar MI/IPU not found")
    message(STATUS "  platform   : ${THIRD_PARTY_PLATFORM_TAG}")
    message(STATUS "  chip arch  : ${SIGMASTAR_CHIP_ARCH}")
    message(STATUS "  expected   : ${_SIGMASTAR_SDK_ROOT}")
    message(STATUS "  include    : ${_SIGMASTAR_IPU_INCLUDE_DIR}/mi_ipu.h")
    message(STATUS "  mi libs    : ${_SIGMASTAR_MI_LIB_DIR}/libmi_ipu.a")
    message(STATUS "  common     : ${_SIGMASTAR_COMMON_LIB_DIR}/libcam_os_wrapper.a")
    message(STATUS "==============================================================================")
    set(SIGMASTAR_CHIP_ARCH "")
    unset(_SIGMASTAR_MI_LIB_REL)
    unset(_SIGMASTAR_COMMON_LIB_REL)
    unset(_SIGMASTAR_CHIP_DEFINE)
    unset(_SIGMASTAR_SDK_ROOT)
    unset(_SIGMASTAR_MI_LIB_DIR)
    unset(_SIGMASTAR_COMMON_LIB_DIR)
    unset(_SIGMASTAR_INCLUDE_DIR)
    unset(_SIGMASTAR_IPU_INCLUDE_DIR)
    return()
endif()

if(TARGET sigmastar::mi)
    set(SIGMASTAR_FOUND TRUE)
    unset(_SIGMASTAR_MI_LIB_REL)
    unset(_SIGMASTAR_COMMON_LIB_REL)
    unset(_SIGMASTAR_CHIP_DEFINE)
    unset(_SIGMASTAR_SDK_ROOT)
    unset(_SIGMASTAR_MI_LIB_DIR)
    unset(_SIGMASTAR_COMMON_LIB_DIR)
    unset(_SIGMASTAR_INCLUDE_DIR)
    unset(_SIGMASTAR_IPU_INCLUDE_DIR)
    return()
endif()

add_library(sigmastar_mi INTERFACE)
add_library(sigmastar::mi ALIAS sigmastar_mi)

target_compile_definitions(sigmastar_mi INTERFACE
    ${_SIGMASTAR_CHIP_DEFINE}
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

# OpenDLA / MI 检测 demo 所需最小依赖（与厂商 build.mk 对齐）
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
set(SIGMASTAR_CHIP_ARCH "${SIGMASTAR_CHIP_ARCH}" CACHE INTERNAL "SGS chip architecture")

# 单摄上屏：头文件在 include/isp 与 include/isp/<arch>，库与 sigmastar::mi 同目录。
# 不并进 sigmastar::mi，避免只跑 IPU 的示例被 ISP/DISP 静态库牵进去。
set(_SIGMASTAR_ISP_INCLUDE_DIR "${_SIGMASTAR_INCLUDE_DIR}/isp")
set(_SIGMASTAR_ISP_CHIP_INCLUDE_DIR
    "${_SIGMASTAR_ISP_INCLUDE_DIR}/${SIGMASTAR_CHIP_ARCH}")
set(_SIGMASTAR_PIPELINE_LIBS_OK TRUE)
foreach(_lib IN ITEMS mi_sensor mi_vif mi_isp mi_scl mi_disp cus3a ispalgo)
    if(NOT EXISTS "${_SIGMASTAR_MI_LIB_DIR}/lib${_lib}.a")
        set(_SIGMASTAR_PIPELINE_LIBS_OK FALSE)
    endif()
endforeach()
if(NOT EXISTS "${_SIGMASTAR_ISP_INCLUDE_DIR}/mi_isp.h"
   OR NOT EXISTS "${_SIGMASTAR_ISP_CHIP_INCLUDE_DIR}/mi_isp_hw_dep_datatype.h")
    set(_SIGMASTAR_PIPELINE_LIBS_OK FALSE)
endif()

set(_SIGMASTAR_PIPELINE_STATUS "not found")
if(_SIGMASTAR_PIPELINE_LIBS_OK AND NOT TARGET sigmastar::sensor_pipeline)
    add_library(sigmastar_sensor_pipeline INTERFACE)
    add_library(sigmastar::sensor_pipeline ALIAS sigmastar_sensor_pipeline)
    target_link_libraries(sigmastar_sensor_pipeline INTERFACE sigmastar::mi)
    target_include_directories(sigmastar_sensor_pipeline INTERFACE
        "${_SIGMASTAR_ISP_INCLUDE_DIR}"
        "${_SIGMASTAR_ISP_CHIP_INCLUDE_DIR}"
    )
    # mi_isp 与 cus3a/ispalgo/mi_sys 互相引用，放进同一组。
    target_link_libraries(sigmastar_sensor_pipeline INTERFACE
        -Wl,--start-group
        mi_sensor
        mi_vif
        mi_isp
        mi_scl
        mi_disp
        cus3a
        ispalgo
        mi_sys
        mi_common
        cam_os_wrapper
        cam_fs_wrapper
        -Wl,--end-group
    )
    set(_SIGMASTAR_PIPELINE_STATUS "sigmastar::sensor_pipeline")
elseif(TARGET sigmastar::sensor_pipeline)
    set(_SIGMASTAR_PIPELINE_STATUS "sigmastar::sensor_pipeline")
endif()

message(STATUS "==============================================================================")
message(STATUS "Sigmastar MI/IPU found")
message(STATUS "  platform  : ${THIRD_PARTY_PLATFORM_TAG}")
message(STATUS "  chip arch : ${SIGMASTAR_CHIP_ARCH} (${_SIGMASTAR_CHIP_DEFINE})")
message(STATUS "  sdk root  : ${_SIGMASTAR_SDK_ROOT}")
message(STATUS "  mi libs   : ${_SIGMASTAR_MI_LIB_DIR}")
message(STATUS "  common    : ${_SIGMASTAR_COMMON_LIB_DIR}")
message(STATUS "  include   : ${_SIGMASTAR_INCLUDE_DIR}")
message(STATUS "  ipu       : ${_SIGMASTAR_IPU_INCLUDE_DIR}")
message(STATUS "  isp       : ${_SIGMASTAR_ISP_INCLUDE_DIR}")
message(STATUS "  target    : sigmastar::mi")
message(STATUS "  pipeline  : ${_SIGMASTAR_PIPELINE_STATUS}")
message(STATUS "==============================================================================")

unset(_SIGMASTAR_ISP_INCLUDE_DIR)
unset(_SIGMASTAR_ISP_CHIP_INCLUDE_DIR)
unset(_SIGMASTAR_PIPELINE_LIBS_OK)
unset(_SIGMASTAR_PIPELINE_STATUS)

unset(_SIGMASTAR_MI_LIB_REL)
unset(_SIGMASTAR_COMMON_LIB_REL)
unset(_SIGMASTAR_CHIP_DEFINE)
unset(_SIGMASTAR_SDK_ROOT)
unset(_SIGMASTAR_MI_LIB_DIR)
unset(_SIGMASTAR_COMMON_LIB_DIR)
unset(_SIGMASTAR_INCLUDE_DIR)
unset(_SIGMASTAR_IPU_INCLUDE_DIR)
