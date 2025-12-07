# FindOpenCVCustom.cmake - 自定义OpenCV库查找模块

# set(OpenCV_DIR ${THIRD_PARTY_DIR}/opencv/${CMAKE_SYSTEM_PROCESSOR}/lib/cmake/opencv4)
list(INSERT CMAKE_PREFIX_PATH 0  "${THIRD_PARTY_DIR}/opencv/${CMAKE_SYSTEM_PROCESSOR}")

find_package(OpenCV REQUIRED)
message(STATUS "==============================================================================")
if(OpenCV_FOUND)
    message(STATUS "OpenCV found successfully")
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
    message(STATUS "OpenCV config file: ${OpenCV_CONFIG}")
else()
    message(FATAL_ERROR "OpenCV not found")
endif()
message(STATUS "==============================================================================")