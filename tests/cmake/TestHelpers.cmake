# add_percept_test - 注册 GTest 可执行文件并加入 ctest
#
# 用法:
#   add_percept_test(
#       NAME test_foo
#       TIER unit                    # smoke | unit | integration
#       SOURCES path/to/test.cpp
#       LINK_LIBS root_sdk_percept   # 可选
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

    install(TARGETS ${PARSED_NAME} RUNTIME DESTINATION bin)

    add_test(NAME ${PARSED_NAME} COMMAND ${PARSED_NAME})
    set_tests_properties(${PARSED_NAME} PROPERTIES
        LABELS "${PARSED_TIER}"
        ENVIRONMENT
            "PERCEPT_ROOT=${CMAKE_SOURCE_DIR};PERCEPT_OUTPUT_DIR=${CMAKE_SOURCE_DIR}/tmp")
endfunction()
