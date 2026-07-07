cmake_minimum_required(VERSION 3.10)

# 定义项目名称
project(deploy_percept VERSION 0.1.0 LANGUAGES CXX C)

option(ENABLE_TESTS "Enable tests (smoke, unit, integration)" ON)
option(INSTALL_TESTS "Install test executables to tests/ (requires ENABLE_TESTS)" ON)
option(ENABLE_BENCHMARKS "Build performance benchmarks" OFF)
option(INSTALL_BENCHMARKS "Install benchmarks to benchmarks/ (requires ENABLE_BENCHMARKS)" ON)

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
if(ENABLE_TESTS)
    enable_testing()
    if(GTest_FOUND AND CNPY_FOUND AND OpenCV_FOUND AND spdlog_FOUND)
        message(STATUS "测试依赖齐备，进入 tests/")
        add_subdirectory(tests)
    else()
        message(WARNING
            "ENABLE_TESTS=ON 但测试依赖未齐备，跳过 tests"
            " (gtest=${GTest_FOUND} cnpy=${CNPY_FOUND} opencv=${OpenCV_FOUND} spdlog=${spdlog_FOUND})")
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
    if(CNPY_FOUND AND OpenCV_FOUND AND spdlog_FOUND)
        message(STATUS "Benchmark utils 依赖齐备，进入 benchmarks/")
        add_subdirectory(benchmarks)
    else()
        message(WARNING
            "ENABLE_BENCHMARKS=ON 但 utils 依赖未齐备，跳过 benchmarks"
            " (cnpy=${CNPY_FOUND} opencv=${OpenCV_FOUND} spdlog=${spdlog_FOUND})")
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

message(STATUS "==============================================================================")
