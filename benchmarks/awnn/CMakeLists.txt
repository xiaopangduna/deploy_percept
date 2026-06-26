if(NOT AWNN_FOUND)
    message(STATUS "benchmarks/awnn: AWNN 未找到，跳过 AWNN benchmark")
    return()
endif()

add_percept_awnn_benchmark(
    NAME bench_yolov5_detect_awnn_mapped_vs_hostcopy
    SOURCES
        bench_output.cpp
        bench_yolov5_detect_awnn_mapped_vs_hostcopy.cpp)
