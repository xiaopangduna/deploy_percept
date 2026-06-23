# FindZlibCustom.cmake
# 使用 third_party 中的 zlib（静态库）
#
# 交叉编译下不用 find_library/find_path，避免 FIND_ROOT_PATH_MODE 导致查找失败

set(ZLIB_ROOT "${THIRD_PARTY_PLATFORM_DIR}/zlib")
get_filename_component(ZLIB_ROOT "${ZLIB_ROOT}" ABSOLUTE)
set(ZLIB_INCLUDE_DIR "${ZLIB_ROOT}/include")
set(ZLIB_LIBRARY "${ZLIB_ROOT}/lib/libz.a")

if(NOT EXISTS "${ZLIB_LIBRARY}" OR NOT EXISTS "${ZLIB_INCLUDE_DIR}/zlib.h")
    message(FATAL_ERROR
        "zlib not found in ${ZLIB_ROOT}\n"
        "Build with:\n"
        "  bash scripts/third_party_builder.sh ${THIRD_PARTY_PLATFORM_TAG} --libs cnpy")
endif()

set(ZlibCustom_FOUND TRUE)

message(STATUS "==============================================================================")
message(STATUS "ZLIB found successfully")
message(STATUS "  ZLIB root        : ${ZLIB_ROOT}")
message(STATUS "  ZLIB include dir : ${ZLIB_INCLUDE_DIR}")
message(STATUS "  ZLIB library     : ${ZLIB_LIBRARY}")
message(STATUS "==============================================================================")

if(NOT TARGET ZLIB::ZLIB)
    add_library(ZLIB::ZLIB STATIC IMPORTED)
    set_target_properties(ZLIB::ZLIB PROPERTIES
        IMPORTED_LOCATION "${ZLIB_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ZLIB_INCLUDE_DIR}"
    )
endif()
