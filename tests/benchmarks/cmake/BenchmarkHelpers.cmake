include(GNUInstallDirs)

# 源码在 tests/benchmarks/；install 后与 bin/、lib/、tests/ 同级
set(PERCEPT_BENCHMARKS_INSTALL_DIR "benchmarks")
# prefix/benchmarks/<exe> → prefix/lib
set(PERCEPT_BENCHMARKS_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")

function(percept_install_benchmark)
    if(NOT INSTALL_BENCHMARKS)
        return()
    endif()
    if(NOT TARGET ${ARGV0})
        message(FATAL_ERROR "percept_install_benchmark: unknown target '${ARGV0}'")
    endif()
    install(TARGETS ${ARGV0} RUNTIME DESTINATION ${PERCEPT_BENCHMARKS_INSTALL_DIR})
    set_target_properties(${ARGV0} PROPERTIES INSTALL_RPATH "${PERCEPT_BENCHMARKS_INSTALL_RPATH}")
endfunction()

# add_percept_awnn_benchmark - AWNN 性能 benchmark
#
# 用法:
#   add_percept_awnn_benchmark(
#       NAME bench_yolov5
#       SOURCES bench_yolov5.cpp
#   )

function(add_percept_awnn_benchmark)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES)
    cmake_parse_arguments(PARSED "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT PARSED_NAME)
        message(FATAL_ERROR "add_percept_awnn_benchmark: NAME is required")
    endif()
    if(NOT AWNN_FOUND)
        message(FATAL_ERROR "add_percept_awnn_benchmark(${PARSED_NAME}): AWNN not found")
    endif()
    if(NOT TARGET test_common)
        message(FATAL_ERROR "add_percept_awnn_benchmark(${PARSED_NAME}): test_common not built")
    endif()
    if(NOT TARGET deploy_percept_utils)
        message(FATAL_ERROR "add_percept_awnn_benchmark(${PARSED_NAME}): deploy_percept_utils not built")
    endif()

    add_executable(${PARSED_NAME} ${PARSED_SOURCES})

    target_include_directories(${PARSED_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/tests
        ${CMAKE_SOURCE_DIR})

    target_link_libraries(${PARSED_NAME} PRIVATE
        test_common
        deploy_percept_core
        deploy_percept_utils
        AWNN::VIPLite
        pthread
        ${OpenCV_LIBS})

    if(OpenCV_INCLUDE_DIRS)
        target_include_directories(${PARSED_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
    endif()

    set_target_properties(${PARSED_NAME} PROPERTIES
        BUILD_RPATH "${AWNN_VIPLITE_LIB_DIR}")

    percept_install_benchmark(${PARSED_NAME})
endfunction()

# add_percept_sigmastar_benchmark - Sigmastar IPU 性能 benchmark
#
# 用法:
#   add_percept_sigmastar_benchmark(
#       NAME bench_ipu_img
#       SOURCES bench_ipu_img.cpp
#   )

function(add_percept_sigmastar_benchmark)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES)
    cmake_parse_arguments(PARSED "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT PARSED_NAME)
        message(FATAL_ERROR "add_percept_sigmastar_benchmark: NAME is required")
    endif()
    if(NOT SIGMASTAR_FOUND)
        message(FATAL_ERROR "add_percept_sigmastar_benchmark(${PARSED_NAME}): SIGMASTAR not found")
    endif()
    if(NOT TARGET test_common)
        message(FATAL_ERROR "add_percept_sigmastar_benchmark(${PARSED_NAME}): test_common not built")
    endif()
    if(NOT TARGET deploy_percept_core)
        message(FATAL_ERROR "add_percept_sigmastar_benchmark(${PARSED_NAME}): deploy_percept_core not built")
    endif()
    if(NOT TARGET sigmastar::mi)
        message(FATAL_ERROR "add_percept_sigmastar_benchmark(${PARSED_NAME}): sigmastar::mi missing")
    endif()

    add_executable(${PARSED_NAME} ${PARSED_SOURCES})

    target_include_directories(${PARSED_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/tests
        ${CMAKE_SOURCE_DIR})

    target_compile_definitions(${PARSED_NAME} PRIVATE
        "PERCEPT_ROOT=\"${CMAKE_SOURCE_DIR}\"")

    target_link_libraries(${PARSED_NAME} PRIVATE
        test_common
        deploy_percept_core
        sigmastar::mi
        ${OpenCV_LIBS})

    if(OpenCV_INCLUDE_DIRS)
        target_include_directories(${PARSED_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
    endif()

    percept_install_benchmark(${PARSED_NAME})
endfunction()
