cmake_minimum_required(VERSION 3.10)

# 定义项目名称
project(deploy_percept VERSION 0.1.0 LANGUAGES CXX C)

option(ENABLE_TESTS "Enable tests (smoke, unit, integration)" ON)
option(BUILD_RKNN_APPS "Build RKNN demo applications (requires RKNN runtime)" OFF)
option(BUILD_AWNN_APPS "Build Allwinner AWNN demo applications (requires ai-sdk VIPLite)" ON)

# 指定C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 工具链/平台变量见各 toolchain 文件；此处仅打印 project() 之后的工程配置
message(STATUS "==============================================================================")
message(STATUS "项目配置:")
message(STATUS "  CMAKE_BUILD_TYPE           = ${CMAKE_BUILD_TYPE}")
message(STATUS "  CMAKE_CXX_STANDARD         = ${CMAKE_CXX_STANDARD}")
message(STATUS "  CMAKE_CROSSCOMPILING       = ${CMAKE_CROSSCOMPILING}")
message(STATUS "  CMAKE_CXX_COMPILER_ID      = ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  CMAKE_CXX_COMPILER_VERSION = ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "==============================================================================")

# cmake/ 放 FindThirdParty；cmake/modules/ 放各库 Find 模块
list(INSERT CMAKE_MODULE_PATH 0
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

# 包含第三方库查找模块
include(FindThirdParty)

# 定义源文件列表
set(SOURCES_FILE
    src/utils/logger_factory.cpp
    src/utils/npy.cpp
    src/utils/io.cpp
    src/pre_process/image_pre_process.cpp
    src/post_process/YoloBasePostProcess.cpp
    src/post_process/YoloV5DetectPostProcess.cpp
    src/post_process/YoloV5SegPostProcess.cpp
    src/post_process/YoloV8PosePostProcess.cpp
    src/post_process/YoloV8SegPostProcess.cpp
)

# 创建库目标
add_library(deploy_percept STATIC ${SOURCES_FILE})

target_link_libraries(deploy_percept PUBLIC
    ${OpenCV_LIBS}
    spdlog::spdlog
    yaml-cpp::yaml-cpp
    cnpy::cnpy
    ZLIB::ZLIB
)

# 设置库的包含目录
target_include_directories(deploy_percept PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Sigmastar MI SDK（armv7l SSC375 平台后端，条件链接）
if(SIGMASTAR_FOUND)
    target_sources(deploy_percept PRIVATE
        src/engine/SigmastarIpuEngine.cpp
    )
    target_link_libraries(deploy_percept PUBLIC sigmastar::mi)
    target_compile_definitions(deploy_percept PUBLIC SIGMASTAR_FOUND=1)
endif()

# 添加apps目录
add_subdirectory(apps)

message(STATUS "==============================================================================")
message(STATUS "测试配置信息:")
message(STATUS "  - ENABLE_TESTS: ${ENABLE_TESTS}")
message(STATUS "==============================================================================")
if(ENABLE_TESTS)
    message(STATUS "测试已被启用，开始编译测试用例")
    enable_testing()
    add_subdirectory(tests)
else()
    message(STATUS "测试已被禁用")
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/PerceptInstall.cmake)
message(STATUS "==============================================================================")