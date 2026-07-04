list(APPEND CMAKE_MODULE_PATH
    "${CMAKE_SOURCE_DIR}/cmake/modules"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(TestHelpers)

add_library(test_common STATIC
    test_common/paths.cpp
    test_common/utils.cpp
    test_common/fixtures.cpp)
target_include_directories(test_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_common PUBLIC
    deploy_percept_utils
    GTest::gtest)

if(INSTALL_TESTS)
    install(
        DIRECTORY "${CMAKE_SOURCE_DIR}/tests/fixtures/"
        DESTINATION share/percept/fixtures
        USE_SOURCE_PERMISSIONS)
endif()

add_subdirectory(smoke)
add_subdirectory(unit)
add_subdirectory(integration)
