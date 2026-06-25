# FindOpenCVCustom.cmake - 项目定制 OpenCV 查找模块
#
# 对外统一暴露: OpenCV_FOUND, OpenCV_LIBS, OpenCV_INCLUDE_DIRS, OpenCV_VERSION, OpenCV_CONFIG
#
# 1. find_package @ ${THIRD_PARTY_PLATFORM_DIR}/opencv
# 2. armv7l-SSC375 失败时回退 SDK 预编译 (OpenCVSigmastar.cmake)

list(INSERT CMAKE_PREFIX_PATH 0 "${THIRD_PARTY_PLATFORM_DIR}/opencv")

find_package(OpenCV QUIET)

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR
        "==============================================================================\n"
        "OpenCV not found: ${THIRD_PARTY_PLATFORM_DIR}/opencv\n"
        "Build with:\n"
        "  bash scripts/third_party_builders/builder_opencv.sh --platform ${THIRD_PARTY_PLATFORM_TAG}\n"
        "==============================================================================")
endif()

message(STATUS "==============================================================================")
message(STATUS "OpenCV found: ${OpenCV_VERSION}")
message(STATUS "  include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "  libs: ${OpenCV_LIBS}")
message(STATUS "==============================================================================")
