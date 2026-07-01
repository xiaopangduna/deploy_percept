# FindGTestCustom.cmake - 自定义 Google Test 库查找模块（非致命）
#
# find_package(GTest) 会设置 GTest_FOUND、GTest::gtest 等，无需再手动定义 *_FOUND。

list(INSERT CMAKE_PREFIX_PATH 0 "${THIRD_PARTY_PLATFORM_DIR}/gtest")

find_package(GTest QUIET)

if(NOT GTest_FOUND)
    message(STATUS "GTest not found under ${THIRD_PARTY_PLATFORM_DIR}/gtest")
    message(STATUS "  build: bash scripts/third_party_builder.sh ${THIRD_PARTY_PLATFORM_TAG} --libs gtest")
    return()
endif()

message(STATUS "==============================================================================")
message(STATUS "GTest found successfully")
message(STATUS "GTest version: ${GTest_VERSION}")
message(STATUS "GTest libraries: ${GTEST_LIBRARIES}")
message(STATUS "GTest config file: ${GTest_CONFIG}")
message(STATUS "==============================================================================")
