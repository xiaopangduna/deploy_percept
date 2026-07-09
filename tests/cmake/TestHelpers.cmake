include(GNUInstallDirs)

# 相对 CMAKE_INSTALL_PREFIX（与 bin/、include/ 同级）
set(PERCEPT_TESTS_INSTALL_DIR "tests")
# prefix/tests/<exe> → prefix/lib
set(PERCEPT_TESTS_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")

function(percept_install_test)
    if(NOT INSTALL_TESTS)
        return()
    endif()
    if(NOT TARGET ${ARGV0})
        message(FATAL_ERROR "percept_install_test: unknown target '${ARGV0}'")
    endif()
    install(TARGETS ${ARGV0} RUNTIME DESTINATION ${PERCEPT_TESTS_INSTALL_DIR})
    set_target_properties(${ARGV0} PROPERTIES INSTALL_RPATH "${PERCEPT_TESTS_INSTALL_RPATH}")
endfunction()

# add_percept_test - 注册 GTest 可执行文件并加入 ctest
#
# 用法:
#   add_percept_test(
#       NAME test_foo
#       TIER unit                    # smoke | unit | integration
#       SOURCES path/to/test.cpp
#       LINK_LIBS deploy_percept_core deploy_percept_utils  # 可选
#       USE_CUSTOM_MAIN              # 可选，源文件自带 main 时设置
#       AWNN                         # 可选，AWNN 集成测试（链 VIPLite + 板端 ctest 禁用）
#   )

function(add_percept_test)
    set(options USE_CUSTOM_MAIN AWNN)
    set(oneValueArgs NAME TIER)
    set(multiValueArgs SOURCES LINK_LIBS)
    cmake_parse_arguments(PARSED "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT PARSED_NAME)
        message(FATAL_ERROR "add_percept_test: NAME is required")
    endif()
    if(NOT PARSED_TIER)
        message(FATAL_ERROR "add_percept_test(${PARSED_NAME}): TIER is required (smoke|unit|integration)")
    endif()
    if(PARSED_AWNN AND NOT AWNN_FOUND)
        message(FATAL_ERROR "add_percept_test(${PARSED_NAME}): AWNN not found")
    endif()

    add_executable(${PARSED_NAME} ${PARSED_SOURCES})

    set(_include_dirs
        ${CMAKE_SOURCE_DIR}/tests
        ${CMAKE_SOURCE_DIR})
    if(PARSED_AWNN)
        list(APPEND _include_dirs ${CMAKE_CURRENT_SOURCE_DIR})
    endif()
    target_include_directories(${PARSED_NAME} PRIVATE ${_include_dirs})

    target_compile_definitions(${PARSED_NAME} PRIVATE
        "PERCEPT_ROOT=\"${CMAKE_SOURCE_DIR}\"")

    if(PARSED_USE_CUSTOM_MAIN)
        target_link_libraries(${PARSED_NAME} PRIVATE
            GTest::gtest test_common_gtest ${PARSED_LINK_LIBS})
    else()
        target_link_libraries(${PARSED_NAME} PRIVATE
            GTest::gtest_main test_common_gtest ${PARSED_LINK_LIBS})
    endif()

    if(PARSED_AWNN)
        target_link_libraries(${PARSED_NAME} PRIVATE AWNN::VIPLite pthread)
        set_target_properties(${PARSED_NAME} PROPERTIES
            BUILD_RPATH "${AWNN_VIPLITE_LIB_DIR}")
    endif()

    percept_install_test(${PARSED_NAME})

    set(_test_labels "${PARSED_TIER}")
    if(PARSED_AWNN)
        set(_test_labels "integration;awnn")
    endif()

    add_test(NAME ${PARSED_NAME} COMMAND ${PARSED_NAME})
    set_tests_properties(${PARSED_NAME} PROPERTIES
        LABELS "${_test_labels}"
        ENVIRONMENT
            "PERCEPT_ROOT=${CMAKE_SOURCE_DIR};PERCEPT_OUTPUT_DIR=${CMAKE_SOURCE_DIR}/tmp")

    if(PARSED_AWNN AND CMAKE_CROSSCOMPILING)
        set_tests_properties(${PARSED_NAME} PROPERTIES DISABLED TRUE)
        message(STATUS "integration/awnn: ${PARSED_NAME} disabled under cross-compiling (run on board via test.sh)")
    endif()
endfunction()

# 兼容别名：AWNN 集成测试
function(add_percept_awnn_integration_test)
    add_percept_test(${ARGN} TIER integration AWNN)
endfunction()
