cmake_minimum_required(VERSION 3.10)

# 定义项目名称
project(deploy_percept VERSION 0.1.0 LANGUAGES CXX C)

option(ENABLE_TESTS "Enable tests (smoke, unit, integration)" ON)
option(INSTALL_TESTS "Install test executables to tests/ (requires ENABLE_TESTS)" ON)
option(ENABLE_BENCHMARKS "Build performance benchmarks" OFF)
option(INSTALL_BENCHMARKS "Install benchmarks to benchmarks/ (requires ENABLE_BENCHMARKS)" ON)
option(ENABLE_PROFILING "Build resource profiling tools (tests/profiling)" ON)
option(INSTALL_PROFILING "Install profiling tools to profiling/ (requires ENABLE_PROFILING)" ON)

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
message(STATUS "  ENABLE_TESTS               = ${ENABLE_TESTS}")
message(STATUS "  INSTALL_TESTS              = ${INSTALL_TESTS}")
message(STATUS "  ENABLE_BENCHMARKS          = ${ENABLE_BENCHMARKS}")
message(STATUS "  INSTALL_BENCHMARKS         = ${INSTALL_BENCHMARKS}")
message(STATUS "  ENABLE_PROFILING           = ${ENABLE_PROFILING}")
message(STATUS "  INSTALL_PROFILING          = ${INSTALL_PROFILING}")
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
message(STATUS "test 与 Benchmark 依赖:")
message(STATUS "  - GTest_FOUND:   ${GTest_FOUND}")
message(STATUS "  - utils deps:    cnpy=${CNPY_FOUND} opencv=${OpenCV_FOUND} spdlog=${spdlog_FOUND}")
message(STATUS "==============================================================================")
if(ENABLE_TESTS OR ENABLE_BENCHMARKS)
    add_subdirectory(tests/test_common)
endif()

if(ENABLE_TESTS)
    enable_testing()
endif()

if(ENABLE_TESTS OR ENABLE_BENCHMARKS OR ENABLE_PROFILING)
    add_subdirectory(tests)
endif()

message(STATUS "==============================================================================")
