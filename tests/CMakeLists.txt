list(APPEND CMAKE_MODULE_PATH
    "${CMAKE_SOURCE_DIR}/cmake/modules"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(TestHelpers)

if(PERCEPT_SMOKE_TESTS_READY)
    add_subdirectory(smoke)
else()
    message(WARNING "smoke tests skipped: GTest not found")
endif()

if(PERCEPT_UNIT_TESTS_READY OR PERCEPT_INTEGRATION_TESTS_READY)
    add_library(percept_test_paths STATIC test_common/paths.cpp)
    target_include_directories(percept_test_paths PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

    add_library(test_utils STATIC test_common/utils.cpp)
    target_include_directories(test_utils PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
        $<TARGET_PROPERTY:GTest::gtest,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(test_utils PUBLIC GTest::gtest)
endif()

if(PERCEPT_UNIT_TESTS_READY)
    add_subdirectory(unit)
else()
    message(WARNING "unit tests skipped: need GTest + deploy_percept_utils (cnpy + OpenCV + spdlog)")
endif()

if(PERCEPT_INTEGRATION_TESTS_READY)
    add_subdirectory(integration)
else()
    message(WARNING "integration tests skipped: need GTest + AWNN + deploy_percept_utils")
endif()
