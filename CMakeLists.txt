cmake_minimum_required(VERSION 3.10)

# 定义项目名称
project(deploy_percept VERSION 0.1.0 LANGUAGES CXX C)

option(ENABLE_TESTS "Enable tests (smoke, unit, integration)" ON)
option(INSTALL_TESTS "Install test executables to share/percept/tests (requires ENABLE_TESTS)" ON)
option(ENABLE_BENCHMARKS "Build performance benchmarks" OFF)
option(INSTALL_BENCHMARKS "Install benchmarks to share/percept/benchmarks (requires ENABLE_BENCHMARKS)" ON)

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

# 库定义见 src/{common,core,utils}/CMakeLists.txt
add_subdirectory(src)

# 添加apps目录
add_subdirectory(apps)

message(STATUS "==============================================================================")
message(STATUS "测试配置信息:")
message(STATUS "  - ENABLE_TESTS:  ${ENABLE_TESTS}")
message(STATUS "  - INSTALL_TESTS: ${INSTALL_TESTS}")
message(STATUS "  - PERCEPT_UTILS_READY: ${PERCEPT_UTILS_READY}")
message(STATUS "  - PERCEPT_SMOKE_TESTS_READY: ${PERCEPT_SMOKE_TESTS_READY}")
message(STATUS "  - PERCEPT_UNIT_TESTS_READY: ${PERCEPT_UNIT_TESTS_READY}")
message(STATUS "  - PERCEPT_INTEGRATION_TESTS_READY: ${PERCEPT_INTEGRATION_TESTS_READY}")
message(STATUS "Benchmark 配置信息:")
message(STATUS "  - ENABLE_BENCHMARKS:  ${ENABLE_BENCHMARKS}")
message(STATUS "  - INSTALL_BENCHMARKS: ${INSTALL_BENCHMARKS}")
message(STATUS "  - PERCEPT_BENCHMARKS_READY: ${PERCEPT_BENCHMARKS_READY}")
message(STATUS "==============================================================================")
if(ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
    if(NOT PERCEPT_SMOKE_TESTS_READY
            AND NOT PERCEPT_UNIT_TESTS_READY
            AND NOT PERCEPT_INTEGRATION_TESTS_READY)
        message(WARNING "ENABLE_TESTS=ON 但 smoke/unit/integration 三层依赖均未齐备，无测试可编译")
        if(INSTALL_TESTS)
            message(WARNING "INSTALL_TESTS=ON 但 tests 未编译，不会安装测试")
        endif()
    endif()
else()
    message(STATUS "测试已被禁用")
    if(INSTALL_TESTS)
        message(WARNING "INSTALL_TESTS=ON 但 ENABLE_TESTS=OFF，不会编译或安装测试")
    endif()
endif()

if(ENABLE_BENCHMARKS)
    if(PERCEPT_BENCHMARKS_READY)
        message(STATUS "Benchmark 已被启用，AWNN 依赖齐备，开始编译 benchmark")
        add_subdirectory(benchmarks)
    else()
        message(WARNING "ENABLE_BENCHMARKS=ON 但 AWNN 或 deploy_percept_utils 未就绪，跳过 benchmarks")
        if(INSTALL_BENCHMARKS)
            message(WARNING "INSTALL_BENCHMARKS=ON 但 benchmarks 未编译，不会安装 benchmark")
        endif()
    endif()
else()
    message(STATUS "Benchmark 已被禁用")
    if(INSTALL_BENCHMARKS)
        message(WARNING "INSTALL_BENCHMARKS=ON 但 ENABLE_BENCHMARKS=OFF，不会编译或安装 benchmark")
    endif()
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/PerceptInstall.cmake)
message(STATUS "==============================================================================")
