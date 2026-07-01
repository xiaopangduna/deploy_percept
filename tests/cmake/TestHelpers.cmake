# 测试可执行文件 install 相对路径（相对 CMAKE_INSTALL_PREFIX）
set(PERCEPT_TESTS_INSTALL_DIR "share/percept/tests")

# install prefix/lib 相对 share/percept/tests/<exe>
set(PERCEPT_TESTS_INSTALL_RPATH "$ORIGIN/../../../lib")

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
#   )

function(add_percept_test)
    set(options USE_CUSTOM_MAIN)
    set(oneValueArgs NAME TIER)
    set(multiValueArgs SOURCES LINK_LIBS)
    cmake_parse_arguments(PARSED "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT PARSED_NAME)
        message(FATAL_ERROR "add_percept_test: NAME is required")
    endif()
    if(NOT PARSED_TIER)
        message(FATAL_ERROR "add_percept_test(${PARSED_NAME}): TIER is required (smoke|unit|integration)")
    endif()

    add_executable(${PARSED_NAME} ${PARSED_SOURCES})

    target_include_directories(${PARSED_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/tests
        ${CMAKE_SOURCE_DIR}
    )

    target_compile_definitions(${PARSED_NAME} PRIVATE
        "PERCEPT_ROOT=\"${CMAKE_SOURCE_DIR}\"")

    if(PARSED_USE_CUSTOM_MAIN)
        target_link_libraries(${PARSED_NAME} PRIVATE
            GTest::gtest percept_test_paths ${PARSED_LINK_LIBS})
    else()
        target_link_libraries(${PARSED_NAME} PRIVATE
            GTest::gtest_main percept_test_paths ${PARSED_LINK_LIBS})
    endif()

    percept_install_test(${PARSED_NAME})

    add_test(NAME ${PARSED_NAME} COMMAND ${PARSED_NAME})
    set_tests_properties(${PARSED_NAME} PROPERTIES
        LABELS "${PARSED_TIER}"
        ENVIRONMENT
            "PERCEPT_ROOT=${CMAKE_SOURCE_DIR};PERCEPT_OUTPUT_DIR=${CMAKE_SOURCE_DIR}/tmp")
endfunction()

# add_percept_awnn_integration_test - AWNN 集成测试（需 VIPLite + .nb 模型）
#
# 用法:
#   add_percept_awnn_integration_test(
#       NAME test_yolov5_detect_awnn
#       SOURCES pipeline.cpp test_yolov5_detect_awnn.cpp
#       LINK_LIBS deploy_percept_core deploy_percept_utils
#       USE_CUSTOM_MAIN
#   )

function(add_percept_awnn_integration_test)
    set(options USE_CUSTOM_MAIN)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES LINK_LIBS)
    cmake_parse_arguments(PARSED "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT PARSED_NAME)
        message(FATAL_ERROR "add_percept_awnn_integration_test: NAME is required")
    endif()
    if(NOT AWNN_FOUND)
        message(FATAL_ERROR "add_percept_awnn_integration_test(${PARSED_NAME}): AWNN not found")
    endif()

    add_executable(${PARSED_NAME} ${PARSED_SOURCES})

    target_include_directories(${PARSED_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/tests
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR})

    target_compile_definitions(${PARSED_NAME} PRIVATE
        "PERCEPT_ROOT=\"${CMAKE_SOURCE_DIR}\"")

    if(PARSED_USE_CUSTOM_MAIN)
        target_link_libraries(${PARSED_NAME} PRIVATE
            GTest::gtest percept_test_paths ${PARSED_LINK_LIBS})
    else()
        target_link_libraries(${PARSED_NAME} PRIVATE
            GTest::gtest_main percept_test_paths ${PARSED_LINK_LIBS})
    endif()

    target_link_libraries(${PARSED_NAME} PRIVATE AWNN::VIPLite pthread)

    set_target_properties(${PARSED_NAME} PROPERTIES
        BUILD_RPATH "${AWNN_VIPLITE_LIB_DIR}")

    percept_install_test(${PARSED_NAME})

    add_test(NAME ${PARSED_NAME} COMMAND ${PARSED_NAME})
    set_tests_properties(${PARSED_NAME} PROPERTIES
        LABELS "integration;awnn"
        ENVIRONMENT
            "PERCEPT_ROOT=${CMAKE_SOURCE_DIR};PERCEPT_OUTPUT_DIR=${CMAKE_SOURCE_DIR}/tmp")

    if(CMAKE_CROSSCOMPILING)
        set_tests_properties(${PARSED_NAME} PROPERTIES DISABLED TRUE)
        message(STATUS "integration/awnn: ${PARSED_NAME} disabled under cross-compiling (run on board via test.sh)")
    endif()
endfunction()
