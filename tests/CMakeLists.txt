list(APPEND CMAKE_MODULE_PATH
    "${CMAKE_SOURCE_DIR}/cmake/modules"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(GTestCustom REQUIRED)
include(TestHelpers)

add_library(percept_test_paths STATIC test_common/paths.cpp)
target_include_directories(percept_test_paths PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(test_utils STATIC test_common/utils.cpp)
target_include_directories(test_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    $<TARGET_PROPERTY:GTest::gtest,INTERFACE_INCLUDE_DIRECTORIES>)

add_subdirectory(smoke)
add_subdirectory(unit)
add_subdirectory(integration)
