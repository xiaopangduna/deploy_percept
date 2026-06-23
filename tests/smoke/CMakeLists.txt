set(SMOKE_SOURCES test_stdlib.cpp)

if(DEFINED THIRD_PARTY_PLATFORM_DIR)
    set(SPDLOG_FIND_REQUIRED OFF)
    include(FindSpdlogCustom)
    if(spdlog_FOUND)
        list(APPEND SMOKE_SOURCES test_spdlog.cpp)
    else()
        message(STATUS "smoke: spdlog 未就绪，跳过 TEST(Smoke, SpdlogLink)")
        message(STATUS "  请先执行: bash scripts/third_party_builder.sh ${THIRD_PARTY_PLATFORM_TAG} --libs spdlog")
    endif()
else()
    message(WARNING "THIRD_PARTY_PLATFORM_DIR 未设置，smoke 测试不包含第三方库用例")
endif()

add_executable(smoke_tests ${SMOKE_SOURCES})
target_link_libraries(smoke_tests PRIVATE GTest::gtest_main)
if(spdlog_FOUND)
    target_link_libraries(smoke_tests PRIVATE spdlog::spdlog)
endif()

install(TARGETS smoke_tests RUNTIME DESTINATION bin)

add_test(NAME smoke_tests COMMAND smoke_tests)
set_tests_properties(smoke_tests PROPERTIES
    LABELS "smoke"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
