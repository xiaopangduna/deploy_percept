if(NOT AWNN_FOUND)
    message(STATUS "benchmarks/awnn: AWNN 未找到，跳过 AWNN benchmark")
    return()
endif()

set(PERCEPT_AWNN_BENCH_COMMON_SOURCES
    bench_common.cpp
    bench_pipeline.cpp
    bench_report.cpp
    bench_run.cpp)

add_percept_awnn_benchmark(
    NAME bench_yolov5_post_process
    SOURCES
        ${PERCEPT_AWNN_BENCH_COMMON_SOURCES}
        bench_yolov5_post_process.cpp)
