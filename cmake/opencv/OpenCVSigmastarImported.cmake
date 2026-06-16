# OpenCVSigmastarImported.cmake
# 基于 SDK OpenCVModules.cmake 的依赖图，IMPORTED_LOCATION 指向 variant 真实路径。
# 不 include SDK 内的 OpenCVModules.cmake，避免其对 release/lib 布局的 EXISTS 检查。

macro(_sigmastar_opencv_import_3rdparty _target _filename)
    if(NOT TARGET ${_target})
        add_library(${_target} STATIC IMPORTED)
        set_target_properties(${_target} PROPERTIES
            IMPORTED_LOCATION "${_OPENCV_3RDPARTY_DIR}/${_filename}"
        )
    endif()
endmacro()

macro(_sigmastar_opencv_import_module _module)
    if(NOT TARGET opencv_${_module})
        add_library(opencv_${_module} STATIC IMPORTED)
        set_target_properties(opencv_${_module} PROPERTIES
            IMPORTED_LOCATION "${_OPENCV_LIB_DIR}/libopencv_${_module}.a"
            INTERFACE_INCLUDE_DIRECTORIES "${_OPENCV_INCLUDE_DIR}"
        )
    endif()
endmacro()

_sigmastar_opencv_import_3rdparty(zlib "libzlib.a")
_sigmastar_opencv_import_3rdparty(libjpeg-turbo "liblibjpeg-turbo.a")
_sigmastar_opencv_import_3rdparty(libtiff "liblibtiff.a")
_sigmastar_opencv_import_3rdparty(libwebp "liblibwebp.a")
_sigmastar_opencv_import_3rdparty(libjasper "liblibjasper.a")
_sigmastar_opencv_import_3rdparty(libpng "liblibpng.a")
_sigmastar_opencv_import_3rdparty(libprotobuf "liblibprotobuf.a")
_sigmastar_opencv_import_3rdparty(quirc "libquirc.a")
_sigmastar_opencv_import_3rdparty(tegra_hal "libtegra_hal.a")
_sigmastar_opencv_import_3rdparty(ittnotify "libittnotify.a")

set_target_properties(libtiff PROPERTIES INTERFACE_LINK_LIBRARIES "zlib")
set_target_properties(libpng PROPERTIES INTERFACE_LINK_LIBRARIES "zlib")
set_target_properties(ittnotify PROPERTIES INTERFACE_LINK_LIBRARIES "dl")

if(NOT TARGET ocv.3rdparty.v4l)
    add_library(ocv.3rdparty.v4l INTERFACE IMPORTED)
    set_target_properties(ocv.3rdparty.v4l PROPERTIES
        INTERFACE_COMPILE_DEFINITIONS "HAVE_CAMV4L2"
    )
endif()

set(_opencv_sys_libs
    "$<LINK_ONLY:dl>;$<LINK_ONLY:m>;$<LINK_ONLY:pthread>;$<LINK_ONLY:rt>;$<LINK_ONLY:tegra_hal>"
)

foreach(_mod
    core flann imgproc ml photo dnn features2d imgcodecs
    videoio calib3d highgui objdetect stitching video
)
    _sigmastar_opencv_import_module(${_mod})
endforeach()

set_target_properties(opencv_core PROPERTIES
    INTERFACE_LINK_LIBRARIES
        "${_opencv_sys_libs};$<LINK_ONLY:zlib>;$<LINK_ONLY:ittnotify>"
)
set_target_properties(opencv_flann PROPERTIES
    INTERFACE_LINK_LIBRARIES "opencv_core;${_opencv_sys_libs}"
)
set_target_properties(opencv_imgproc PROPERTIES
    INTERFACE_LINK_LIBRARIES "opencv_core;${_opencv_sys_libs}"
)
set_target_properties(opencv_ml PROPERTIES
    INTERFACE_LINK_LIBRARIES "opencv_core;${_opencv_sys_libs}"
)
set_target_properties(opencv_photo PROPERTIES
    INTERFACE_LINK_LIBRARIES "opencv_core;opencv_imgproc;${_opencv_sys_libs}"
)
set_target_properties(opencv_dnn PROPERTIES
    INTERFACE_LINK_LIBRARIES
        "opencv_core;opencv_imgproc;${_opencv_sys_libs};$<LINK_ONLY:libprotobuf>"
)
set_target_properties(opencv_features2d PROPERTIES
    INTERFACE_LINK_LIBRARIES
        "opencv_core;opencv_flann;opencv_imgproc;${_opencv_sys_libs}"
)
set_target_properties(opencv_imgcodecs PROPERTIES
    INTERFACE_LINK_LIBRARIES
    "opencv_core;opencv_imgproc;${_opencv_sys_libs};$<LINK_ONLY:zlib>;$<LINK_ONLY:libjpeg-turbo>;$<LINK_ONLY:libwebp>;$<LINK_ONLY:libpng>;$<LINK_ONLY:libtiff>;$<LINK_ONLY:libjasper>"
)
set_target_properties(opencv_videoio PROPERTIES
    INTERFACE_LINK_LIBRARIES
    "opencv_core;opencv_imgproc;opencv_imgcodecs;${_opencv_sys_libs};$<LINK_ONLY:ocv.3rdparty.v4l>"
)
set_target_properties(opencv_calib3d PROPERTIES
    INTERFACE_LINK_LIBRARIES
        "opencv_core;opencv_flann;opencv_imgproc;opencv_features2d;${_opencv_sys_libs}"
)
set_target_properties(opencv_highgui PROPERTIES
    INTERFACE_LINK_LIBRARIES
        "opencv_core;opencv_imgproc;opencv_imgcodecs;opencv_videoio;${_opencv_sys_libs}"
)
set_target_properties(opencv_objdetect PROPERTIES
    INTERFACE_LINK_LIBRARIES
    "opencv_core;opencv_flann;opencv_imgproc;opencv_features2d;opencv_calib3d;${_opencv_sys_libs};$<LINK_ONLY:quirc>"
)
set_target_properties(opencv_stitching PROPERTIES
    INTERFACE_LINK_LIBRARIES
    "opencv_core;opencv_flann;opencv_imgproc;opencv_features2d;opencv_calib3d;${_opencv_sys_libs}"
)
set_target_properties(opencv_video PROPERTIES
    INTERFACE_LINK_LIBRARIES
    "opencv_core;opencv_flann;opencv_imgproc;opencv_features2d;opencv_calib3d;${_opencv_sys_libs}"
)

unset(_opencv_sys_libs)
