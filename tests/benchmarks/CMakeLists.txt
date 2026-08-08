list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(BenchmarkHelpers)

# 子目录按引擎划分；各子目录自行用 *_FOUND 决定是否编译
if(AWNN_FOUND)
    add_subdirectory(awnn)
else()
    message(STATUS "tests/benchmarks/awnn: AWNN 未找到，跳过 AWNN benchmark")
endif()

if(SIGMASTAR_FOUND)
    add_subdirectory(sigmastar)
else()
    message(STATUS "tests/benchmarks/sigmastar: SIGMASTAR 未找到，跳过 Sigmastar benchmark")
endif()
