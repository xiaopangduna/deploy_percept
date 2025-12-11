# FindSpdlogCustom.cmake - 自定义spdlog库查找模块

# 推荐方式：直接设置spdlog_DIR变量指向配置文件目录
# set(spdlog_DIR ${THIRD_PARTY_DIR}/spdlog/${CMAKE_SYSTEM_PROCESSOR}/lib/cmake/spdlog)
# 备选方式：将spdlog路径添加到CMAKE_PREFIX_PATH中，让CMake自动查找
list(INSERT CMAKE_PREFIX_PATH 0 "${THIRD_PARTY_DIR}/spdlog/${CMAKE_SYSTEM_PROCESSOR}")

# 使用find_package查找spdlog（Config模式）
find_package(spdlog REQUIRED)

message(STATUS "==============================================================================")
if(spdlog_FOUND)
    message(STATUS "spdlog found successfully")
    message(STATUS "spdlog version: ${spdlog_VERSION}")
    message(STATUS "spdlog include dirs: ${spdlog_INCLUDE_DIRS}")
    message(STATUS "spdlog libraries: ${spdlog_LIBRARIES}")
    message(STATUS "spdlog config file: ${spdlog_CONFIG}")
else()
    message(FATAL_ERROR "spdlog not found")
endif()
message(STATUS "==============================================================================")