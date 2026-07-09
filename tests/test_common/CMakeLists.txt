add_library(test_common STATIC
    paths.cpp
    fixtures.cpp)

target_include_directories(test_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(test_common PUBLIC deploy_percept_common)

if(ENABLE_TESTS AND GTest_FOUND)
    add_library(test_common_gtest STATIC utils.cpp)
    target_include_directories(test_common_gtest PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(test_common_gtest PUBLIC test_common GTest::gtest)
endif()
