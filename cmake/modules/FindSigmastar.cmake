# FindSigmastar.cmake - Sigmastar MI SDK (armv7l-SSC375)

if(NOT THIRD_PARTY_PLATFORM_TAG STREQUAL "armv7l-SSC375")
    message(FATAL_ERROR
        "FindSigmastar: unsupported platform '${THIRD_PARTY_PLATFORM_TAG}' (only armv7l-SSC375)")
endif()

set(_SIGMASTAR_SDK_ROOT "${THIRD_PARTY_PLATFORM_DIR}/project/release")
set(_SIGMASTAR_MI_LIB_DIR
    "${_SIGMASTAR_SDK_ROOT}/chip/ifado/ipc/common/uclibc/9.1.0/release/mi_libs/static"
)
set(_SIGMASTAR_COMMON_LIB_DIR
    "${_SIGMASTAR_SDK_ROOT}/chip/ifado/sigma_common_libs/uclibc/9.1.0/release/static"
)
set(_SIGMASTAR_INCLUDE_DIR "${_SIGMASTAR_SDK_ROOT}/include")
set(_SIGMASTAR_IPU_INCLUDE_DIR "${_SIGMASTAR_INCLUDE_DIR}/ipu/ifado")

if(NOT EXISTS "${_SIGMASTAR_MI_LIB_DIR}")
    message(FATAL_ERROR
        "==============================================================================\n"
        "Sigmastar MI libs not found:\n"
        "  ${_SIGMASTAR_MI_LIB_DIR}\n"
        "==============================================================================")
endif()

if(NOT TARGET sigmastar::mi)
    add_library(sigmastar_mi INTERFACE)
    add_library(sigmastar::mi ALIAS sigmastar_mi)

    target_compile_definitions(sigmastar_mi INTERFACE SGS_308QE)
    target_include_directories(sigmastar_mi INTERFACE
        "${_SIGMASTAR_INCLUDE_DIR}"
        "${_SIGMASTAR_IPU_INCLUDE_DIR}"
    )
    target_link_directories(sigmastar_mi INTERFACE
        "${_SIGMASTAR_MI_LIB_DIR}"
        "${_SIGMASTAR_COMMON_LIB_DIR}"
    )

    target_link_libraries(sigmastar_mi INTERFACE
        -Wl,--start-group
        mi_sys mi_common
        mi_isp ispalgo cus3a
        mi_vif mi_scl mi_venc mi_ipu mi_ive
        mi_ai mi_ao mi_sensor mi_rgn mi_shadow
        mi_vdf mi_vdisp mi_iqserver mi_dummy
        cam_fs_wrapper cam_os_wrapper
        -Wl,--end-group
        pthread m dl rt
    )
endif()

set(SIGMASTAR_FOUND TRUE)

message(STATUS "==============================================================================")
message(STATUS "Sigmastar SDK found: ${THIRD_PARTY_PLATFORM_TAG}")
message(STATUS "  sdk root: ${_SIGMASTAR_SDK_ROOT}")
message(STATUS "  target: sigmastar::mi")
message(STATUS "==============================================================================")

unset(_SIGMASTAR_SDK_ROOT)
unset(_SIGMASTAR_MI_LIB_DIR)
unset(_SIGMASTAR_COMMON_LIB_DIR)
unset(_SIGMASTAR_INCLUDE_DIR)
unset(_SIGMASTAR_IPU_INCLUDE_DIR)
