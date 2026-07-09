# FindGTestCustom.cmake - 自定义 Google Test 库查找模块（非致命）
#
# find_package(GTest) 会设置 GTest_FOUND、GTest::gtest 等。
# third_party 为 shared 时（SHARED IMPORTED），自动 install 运行时 .so 到 CMAKE_INSTALL_LIBDIR。

list(INSERT CMAKE_PREFIX_PATH 0 "${THIRD_PARTY_PLATFORM_DIR}/gtest")

find_package(GTest QUIET)

if(NOT GTest_FOUND)
    message(STATUS "GTest not found under ${THIRD_PARTY_PLATFORM_DIR}/gtest")
    message(STATUS "  build: bash scripts/third_party_builder.sh ${THIRD_PARTY_PLATFORM_TAG} --libs gtest")
    return()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/PerceptThirdPartyInstall.cmake)
percept_install_imported_shared_runtime_targets(GTest::gtest GTest::gtest_main)

message(STATUS "==============================================================================")
message(STATUS "GTest found successfully")
message(STATUS "GTest version: ${GTest_VERSION}")
message(STATUS "GTest libraries: ${GTEST_LIBRARIES}")
message(STATUS "GTest config file: ${GTest_CONFIG}")
message(STATUS "==============================================================================")
