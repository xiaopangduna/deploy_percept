add_executable(smoke_test_stdlib main_test_stdlib.cpp)
install(TARGETS smoke_test_stdlib RUNTIME DESTINATION bin)

if(NOT DEFINED THIRD_PARTY_PLATFORM_DIR)
    message(WARNING "THIRD_PARTY_PLATFORM_DIR 未设置，跳过 smoke_test_third_party")
    return()
endif()

list(INSERT CMAKE_PREFIX_PATH 0
    "${THIRD_PARTY_PLATFORM_DIR}/spdlog"
    "${THIRD_PARTY_PLATFORM_DIR}/yaml-cpp")

find_package(spdlog CONFIG QUIET)
find_package(yaml-cpp CONFIG QUIET)

if(spdlog_FOUND AND yaml-cpp_FOUND)
    add_executable(smoke_test_third_party main_test_third_party.cpp)
    target_link_libraries(smoke_test_third_party PRIVATE spdlog::spdlog yaml-cpp::yaml-cpp)
    install(TARGETS smoke_test_third_party RUNTIME DESTINATION bin)
    message(STATUS "smoke_test: 将构建 smoke_test_third_party (spdlog ${spdlog_VERSION}, yaml-cpp ${yaml-cpp_VERSION})")
else()
    message(STATUS "smoke_test: 跳过 smoke_test_third_party（第三方库未就绪）")
    message(STATUS "  请先执行: bash scripts/third_party_builder.sh ${THIRD_PARTY_PLATFORM_TAG} --libs spdlog,yaml-cpp")
endif()
