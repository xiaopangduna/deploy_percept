# FindGTestCustom.cmake - 自定义Google Test库查找模块

# 推荐方式：直接设置GTest_DIR变量指向配置文件目录
set(GTest_DIR ${THIRD_PARTY_DIR}/gtest/${CMAKE_SYSTEM_PROCESSOR}/lib/cmake/GTest)
# 备选方式：将GTest路径添加到CMAKE_PREFIX_PATH中，让CMake自动查找
# list(INSERT CMAKE_PREFIX_PATH 0 "${THIRD_PARTY_DIR}/gtest/${CMAKE_SYSTEM_PROCESSOR}")

# 使用find_package查找GTest（Config模式）
find_package(GTest REQUIRED)

message(STATUS "==============================================================================")
if(GTest_FOUND)
    message(STATUS "GTest found successfully")
    message(STATUS "GTest version: ${GTest_VERSION}")
    message(STATUS "GTest libraries: ${GTEST_LIBRARIES}")
    message(STATUS "GTest config file: ${GTest_CONFIG}")
else()
    message(FATAL_ERROR "GTest not found")
endif()
message(STATUS "==============================================================================")