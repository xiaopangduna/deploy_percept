# FindZlibCustom.cmake
# 使用 third_party 中的 zlib（静态库）

set(ZLIB_ROOT
    "${THIRD_PARTY_DIR}/zlib/${CMAKE_SYSTEM_PROCESSOR}"
)

# 头文件
find_path(ZLIB_INCLUDE_DIR
    NAMES zlib.h
    PATHS
        ${ZLIB_ROOT}/include
    NO_DEFAULT_PATH
)

# 静态库（优先）
find_library(ZLIB_LIBRARY
    NAMES z
    PATHS
        ${ZLIB_ROOT}/lib
    NO_DEFAULT_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ZlibCustom
    REQUIRED_VARS
        ZLIB_LIBRARY
        ZLIB_INCLUDE_DIR
)

if(ZlibCustom_FOUND)
    message(STATUS "==============================================================================")
    message(STATUS "ZLIB found successfully")
    message(STATUS "ZLIB include dir : ${ZLIB_INCLUDE_DIR}")
    message(STATUS "ZLIB library     : ${ZLIB_LIBRARY}")
    message(STATUS "ZLIB root        : ${ZLIB_ROOT}")
    message(STATUS "==============================================================================")

    if(NOT TARGET ZLIB::ZLIB)
        add_library(ZLIB::ZLIB STATIC IMPORTED)
        set_target_properties(ZLIB::ZLIB PROPERTIES
            IMPORTED_LOCATION "${ZLIB_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${ZLIB_INCLUDE_DIR}"
        )
    endif()
endif()
