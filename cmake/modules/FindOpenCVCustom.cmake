# FindOpenCVCustom.cmake - 项目定制 OpenCV 查找模块
#
# 对外统一暴露: OpenCV_FOUND, OpenCV_LIBS, OpenCV_INCLUDE_DIRS, OpenCV_VERSION, OpenCV_CONFIG

list(INSERT CMAKE_PREFIX_PATH 0 "${THIRD_PARTY_PLATFORM_DIR}/opencv")

find_package(OpenCV QUIET)

if(NOT OpenCV_FOUND)
    message(STATUS "OpenCV not found: ${THIRD_PARTY_PLATFORM_DIR}/opencv")
    message(STATUS "  build: bash scripts/third_party_builder.sh ${THIRD_PARTY_PLATFORM_TAG} --libs opencv")
    return()
endif()

message(STATUS "==============================================================================")
message(STATUS "OpenCV found: ${OpenCV_VERSION}")
message(STATUS "  include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "  libs: ${OpenCV_LIBS}")
message(STATUS "==============================================================================")
