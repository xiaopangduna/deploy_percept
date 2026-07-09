# FindZlibCustom.cmake - 自定义 ZLIB 库查找模块
#
# zlib 上游 configure 默认同时安装 libz.a 与 libz.so；按 ZLIB_LINK_TYPE 拼固定路径并校验，
# 不使用 find_library。交叉编译下避免误链系统或 sysroot 库。

set(ZLIB_FOUND FALSE)

# set(ZLIB_LINK_TYPE "STATIC" CACHE STRING "zlib library link type: STATIC or SHARED")
set(ZLIB_LINK_TYPE "SHARED" CACHE STRING "zlib library link type: STATIC or SHARED")
set_property(CACHE ZLIB_LINK_TYPE PROPERTY STRINGS STATIC SHARED)

set(ZLIB_ROOT "${THIRD_PARTY_PLATFORM_DIR}/zlib")
get_filename_component(ZLIB_ROOT "${ZLIB_ROOT}" ABSOLUTE)
set(ZLIB_INCLUDE_DIR "${ZLIB_ROOT}/include")

if(ZLIB_LINK_TYPE STREQUAL "SHARED")
    set(_ZLIB_LIB_FILE "libz.so")
    set(_ZLIB_IMPORT_TYPE SHARED)
else()
    set(_ZLIB_LIB_FILE "libz.a")
    set(_ZLIB_IMPORT_TYPE STATIC)
endif()

set(ZLIB_LIBRARY "${ZLIB_ROOT}/lib/${_ZLIB_LIB_FILE}")

if(NOT EXISTS "${ZLIB_INCLUDE_DIR}/zlib.h")
    message(STATUS "zlib: missing header ${ZLIB_INCLUDE_DIR}/zlib.h")
    message(STATUS "  build: bash scripts/third_party_builder.sh ${THIRD_PARTY_PLATFORM_TAG} --libs zlib")
    return()
endif()

if(NOT EXISTS "${ZLIB_LIBRARY}")
    message(STATUS "zlib: missing ${ZLIB_LIBRARY} (ZLIB_LINK_TYPE=${ZLIB_LINK_TYPE})")
    message(STATUS "  build: bash scripts/third_party_builder.sh ${THIRD_PARTY_PLATFORM_TAG} --libs zlib")
    return()
endif()

set(ZLIB_FOUND TRUE)

if(NOT TARGET ZLIB::ZLIB)
    add_library(ZLIB::ZLIB ${_ZLIB_IMPORT_TYPE} IMPORTED)
    set_target_properties(ZLIB::ZLIB PROPERTIES
        IMPORTED_LOCATION "${ZLIB_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ZLIB_INCLUDE_DIR}"
    )
endif()

if(ZLIB_LINK_TYPE STREQUAL "SHARED")
    include(${CMAKE_CURRENT_LIST_DIR}/PerceptThirdPartyInstall.cmake)
    percept_install_shared_lib_family("${ZLIB_LIBRARY}")
endif()

message(STATUS "==============================================================================")
message(STATUS "ZLIB found successfully")
message(STATUS "  ZLIB root        : ${ZLIB_ROOT}")
message(STATUS "  ZLIB include dir : ${ZLIB_INCLUDE_DIR}")
message(STATUS "  ZLIB library     : ${ZLIB_LIBRARY}")
message(STATUS "  ZLIB link type   : ${ZLIB_LINK_TYPE}")
message(STATUS "  ZLIB target      : ZLIB::ZLIB")
message(STATUS "==============================================================================")
