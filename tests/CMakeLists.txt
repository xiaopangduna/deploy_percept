list(APPEND CMAKE_MODULE_PATH
    "${CMAKE_SOURCE_DIR}/cmake/modules"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(TestHelpers)
include(InstallFixtures)

add_library(percept_test_paths STATIC test_common/paths.cpp)
target_include_directories(percept_test_paths PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(percept_test_paths PUBLIC deploy_percept_utils)

add_library(test_utils STATIC test_common/utils.cpp)
target_include_directories(test_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    $<TARGET_PROPERTY:GTest::gtest,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(test_utils PUBLIC GTest::gtest)

add_subdirectory(smoke)
add_subdirectory(unit)
add_subdirectory(integration)
