# OpenCVSigmastar.cmake - SSC375 预编译 OpenCV (uclibc static 9.1.0)
#
# 由 FindOpenCVCustom.cmake 在 armv7l-SSC375 且 find_package 失败时 include。
# 不修改 SDK 目录，不依赖 SDK 内的 OpenCVConfig/OpenCVModules。

set(_OPENCV_PREBUILD_ROOT "${THIRD_PARTY_PLATFORM_DIR}/prebuild_libs/opencv")
set(_OPENCV_VARIANT_DIR "${_OPENCV_PREBUILD_ROOT}/release/uclibc_static_lib_9.1.0")
set(_OPENCV_INCLUDE_DIR "${_OPENCV_PREBUILD_ROOT}/include/opencv4")
set(_OPENCV_LIB_DIR "${_OPENCV_VARIANT_DIR}")
set(_OPENCV_3RDPARTY_DIR "${_OPENCV_VARIANT_DIR}/opencv4/3rdparty")

if(NOT EXISTS "${_OPENCV_LIB_DIR}/libopencv_core.a")
    message(FATAL_ERROR "OpenCV libs not found: ${_OPENCV_LIB_DIR}")
endif()
if(NOT EXISTS "${_OPENCV_INCLUDE_DIR}/opencv2/core.hpp")
    message(FATAL_ERROR "OpenCV headers not found: ${_OPENCV_INCLUDE_DIR}")
endif()

if(NOT TARGET opencv_core)
    include("${CMAKE_CURRENT_LIST_DIR}/OpenCVSigmastarImported.cmake")
endif()

set(OpenCV_VERSION 4.1.1)
set(OpenCV_VERSION_MAJOR 4)
set(OpenCV_VERSION_MINOR 1)
set(OpenCV_VERSION_PATCH 1)
set(OpenCV_SHARED OFF)
set(OpenCV_INCLUDE_DIRS "${_OPENCV_INCLUDE_DIR}")

if(NOT OpenCV_FIND_COMPONENTS)
    set(OpenCV_FIND_COMPONENTS core imgproc imgcodecs)
endif()

set(OpenCV_LIBS "")
foreach(_component ${OpenCV_FIND_COMPONENTS})
    if(NOT TARGET opencv_${_component})
        message(FATAL_ERROR
            "OpenCVSigmastar: unknown component '${_component}' (not built in Sigmastar prebuild)")
    endif()
    list(APPEND OpenCV_LIBS opencv_${_component})
    set(OpenCV_${_component}_FOUND TRUE)
endforeach()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenCV
    REQUIRED_VARS OpenCV_INCLUDE_DIRS
    VERSION_VAR OpenCV_VERSION
    HANDLE_COMPONENTS
)

set(OpenCV_CONFIG "${_OPENCV_VARIANT_DIR}/cmake/opencv4")

unset(_OPENCV_PREBUILD_ROOT)
unset(_OPENCV_VARIANT_DIR)
unset(_OPENCV_INCLUDE_DIR)
unset(_OPENCV_LIB_DIR)
unset(_OPENCV_3RDPARTY_DIR)
