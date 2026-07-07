# FindCnpyCustom.cmake - 自定义 Cnpy 库查找模块（非致命）
#
# cnpy 上游会同时安装 libcnpy.a 与 libcnpy.so；按 CNPY_LINK_TYPE 拼固定路径并校验，
# 不使用 find_library。ZLIB 须已由 FindThirdParty.cmake 预先 include(FindZlibCustom)。

set(CNPY_FOUND FALSE)

set(CNPY_LINK_TYPE "STATIC" CACHE STRING "cnpy library link type: STATIC or SHARED")
set_property(CACHE CNPY_LINK_TYPE PROPERTY STRINGS STATIC SHARED)

set(CNPY_PATH "${THIRD_PARTY_PLATFORM_DIR}/cnpy")
set(CNPY_INCLUDE_DIR "${CNPY_PATH}/include")

if(CNPY_LINK_TYPE STREQUAL "SHARED")
    set(_CNPY_LIB_FILE "libcnpy.so")
    set(_CNPY_IMPORT_TYPE SHARED)
else()
    set(_CNPY_LIB_FILE "libcnpy.a")
    set(_CNPY_IMPORT_TYPE STATIC)
endif()

set(CNPY_LIBRARY "${CNPY_PATH}/lib/${_CNPY_LIB_FILE}")

if(NOT EXISTS "${CNPY_INCLUDE_DIR}/cnpy.h")
    message(STATUS "cnpy: missing header ${CNPY_INCLUDE_DIR}/cnpy.h")
    message(STATUS "  build: bash scripts/third_party_builder.sh ${THIRD_PARTY_PLATFORM_TAG} --libs cnpy")
    return()
endif()

if(NOT EXISTS "${CNPY_LIBRARY}")
    message(STATUS "cnpy: missing ${CNPY_LIBRARY} (CNPY_LINK_TYPE=${CNPY_LINK_TYPE})")
    message(STATUS "  build: bash scripts/third_party_builder.sh ${THIRD_PARTY_PLATFORM_TAG} --libs cnpy")
    return()
endif()

if(NOT ZLIB_FOUND OR NOT TARGET ZLIB::ZLIB)
    message(STATUS "cnpy: skipped, ZLIB not available (build zlib first)")
    return()
endif()

if(NOT TARGET cnpy::cnpy)
    add_library(cnpy::cnpy ${_CNPY_IMPORT_TYPE} IMPORTED)
    set_target_properties(cnpy::cnpy PROPERTIES
        IMPORTED_LOCATION "${CNPY_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CNPY_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "ZLIB::ZLIB"
    )
endif()

set(CNPY_FOUND TRUE)
set(CNPY_INCLUDE_DIRS "${CNPY_INCLUDE_DIR}")
set(CNPY_LIBRARIES cnpy::cnpy)

message(STATUS "==============================================================================")
message(STATUS "cnpy found successfully")
message(STATUS "  path    : ${CNPY_PATH}")
message(STATUS "  include : ${CNPY_INCLUDE_DIR}")
message(STATUS "  library : ${CNPY_LIBRARY}")
message(STATUS "  type    : ${CNPY_LINK_TYPE}")
message(STATUS "  target  : cnpy::cnpy")
message(STATUS "==============================================================================")
