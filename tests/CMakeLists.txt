list(APPEND CMAKE_MODULE_PATH
    "${CMAKE_SOURCE_DIR}/cmake/modules"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(ENABLE_TESTS)
    if(GTest_FOUND AND CNPY_FOUND AND OpenCV_FOUND AND spdlog_FOUND)
        include(TestHelpers)

        add_subdirectory(smoke)
        add_subdirectory(unit)
        add_subdirectory(integration)

        if(INSTALL_TESTS)
            include(GNUInstallDirs)
            install(
                DIRECTORY "${CMAKE_SOURCE_DIR}/tests/fixtures/"
                DESTINATION "${CMAKE_INSTALL_DATADIR}/percept/fixtures"
                USE_SOURCE_PERMISSIONS)
            install(
                PROGRAMS "${CMAKE_SOURCE_DIR}/scripts/test_runner.sh"
                DESTINATION scripts)
        endif()
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
        message(STATUS "Benchmark utils 依赖齐备，进入 tests/benchmarks/")
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
