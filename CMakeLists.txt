cmake_minimum_required(VERSION 3.10)

# 定义项目名称
project(deploy_percept VERSION 0.1.0 LANGUAGES CXX C)

option(ENABLE_TESTS "Enable tests (smoke, unit, integration)" ON)
option(INSTALL_TESTS "Install test executables to share/percept/tests (requires ENABLE_TESTS)" ON)
option(ENABLE_BENCHMARKS "Build performance benchmarks" OFF)
option(INSTALL_BENCHMARKS "Install benchmarks to share/percept/benchmarks (requires ENABLE_BENCHMARKS)" ON)
option(BUILD_RKNN_APPS "Build RKNN demo applications (requires RKNN runtime)" OFF)
option(BUILD_AWNN_APPS "Build Allwinner AWNN demo applications (requires VIPLite runtime)" ON)

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
    src/engine/BaseEngine.cpp
    src/engine/OutputAccess.cpp
    src/pre_process/image_pre_process.cpp
    src/post_process/YoloBasePostProcess.cpp
    src/post_process/YoloV5DetectPostProcess.cpp
    src/post_process/YoloV5DetectPostProcessAwnn.cpp
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

# 设置库的包含目录（OpenCV：public 头文件依赖 opencv2/*，需向消费者传播）
target_include_directories(deploy_percept PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>
)

# Sigmastar MI SDK（armv7l SSC375 平台后端，条件链接）
if(SIGMASTAR_FOUND)
    target_sources(deploy_percept PRIVATE
        src/engine/SigmastarIpuEngine.cpp
    )
    target_link_libraries(deploy_percept PUBLIC sigmastar::mi)
    target_compile_definitions(deploy_percept PUBLIC SIGMASTAR_FOUND=1)
endif()

if(AWNN_FOUND)
    target_sources(deploy_percept PRIVATE
        src/engine/AwnnEngine.cpp
    )
    target_link_libraries(deploy_percept PUBLIC AWNN::VIPLite pthread)
    target_compile_definitions(deploy_percept PUBLIC AWNN_FOUND=1)
endif()

# 添加apps目录
add_subdirectory(apps)

message(STATUS "==============================================================================")
message(STATUS "测试配置信息:")
message(STATUS "  - ENABLE_TESTS:  ${ENABLE_TESTS}")
message(STATUS "  - INSTALL_TESTS: ${INSTALL_TESTS}")
message(STATUS "Benchmark 配置信息:")
message(STATUS "  - ENABLE_BENCHMARKS:  ${ENABLE_BENCHMARKS}")
message(STATUS "  - INSTALL_BENCHMARKS: ${INSTALL_BENCHMARKS}")
message(STATUS "==============================================================================")
if(ENABLE_TESTS)
    message(STATUS "测试已被启用，开始编译测试用例")
    enable_testing()
    add_subdirectory(tests)
else()
    message(STATUS "测试已被禁用")
    if(INSTALL_TESTS)
        message(WARNING "INSTALL_TESTS=ON 但 ENABLE_TESTS=OFF，不会编译或安装测试")
    endif()
endif()

if(ENABLE_BENCHMARKS)
    message(STATUS "Benchmark 已被启用，开始编译 benchmark")
    add_subdirectory(benchmarks)
else()
    message(STATUS "Benchmark 已被禁用")
    if(INSTALL_BENCHMARKS)
        message(WARNING "INSTALL_BENCHMARKS=ON 但 ENABLE_BENCHMARKS=OFF，不会编译或安装 benchmark")
    endif()
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/PerceptInstall.cmake)
message(STATUS "==============================================================================")