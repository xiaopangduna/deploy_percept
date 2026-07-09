# FindOpenCVCustom.cmake - 项目定制 OpenCV 查找模块
#
# 对外统一暴露: OpenCV_FOUND, OpenCV_LIBS, OpenCV_INCLUDE_DIRS, OpenCV_VERSION, OpenCV_CONFIG
# OpenCV_SHARED 时，对 OpenCV_LIBS 中的 SHARED target 自动 install 运行时 .so。

list(INSERT CMAKE_PREFIX_PATH 0 "${THIRD_PARTY_PLATFORM_DIR}/opencv")

find_package(OpenCV QUIET)

if(NOT OpenCV_FOUND)
    message(STATUS "OpenCV not found: ${THIRD_PARTY_PLATFORM_DIR}/opencv")
    message(STATUS "  build: bash scripts/third_party_builder.sh ${THIRD_PARTY_PLATFORM_TAG} --libs opencv")
    return()
endif()

if(OpenCV_SHARED)
    include(${CMAKE_CURRENT_LIST_DIR}/PerceptThirdPartyInstall.cmake)
    foreach(_ocv IN LISTS OpenCV_LIBS)
        if(TARGET ${_ocv})
            percept_install_imported_shared_runtime(${_ocv})
        elseif(_ocv MATCHES "\\.so")
            percept_install_shared_lib_family("${_ocv}")
        endif()
    endforeach()
endif()

message(STATUS "==============================================================================")
message(STATUS "OpenCV found: ${OpenCV_VERSION}")
message(STATUS "  shared  : ${OpenCV_SHARED}")
message(STATUS "  include : ${OpenCV_INCLUDE_DIRS}")
message(STATUS "  libs    : ${OpenCV_LIBS}")
message(STATUS "==============================================================================")
