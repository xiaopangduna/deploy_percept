cmake_minimum_required(VERSION 3.10)

# 在测试目录中查找GTest
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")

# 使用自定义的查找宏
find_package(GTestCustom REQUIRED)

# 查找并包含当前目录下的所有测试源文件
file(GLOB TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp)

# 为每个测试源文件创建一个测试目标
foreach(test_file ${TEST_SOURCES})
    get_filename_component(test_name ${test_file} NAME_WE)
    add_executable(${test_name} ${test_file})
    target_link_libraries(${test_name} deploy_percept gtest_main gtest gmock gmock_main)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

enable_testing()