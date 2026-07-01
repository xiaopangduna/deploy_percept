# FindCnpyCustom.cmake - 自定义 Cnpy 库查找模块（非致命）

set(CNPY_FOUND FALSE)

set(CNPY_PATH "${THIRD_PARTY_PLATFORM_DIR}/cnpy")
set(CNPY_INCLUDE_DIR "${CNPY_PATH}/include")
set(CNPY_LIBRARY "${CNPY_PATH}/lib/libcnpy.a")

if(NOT EXISTS "${CNPY_LIBRARY}" OR NOT EXISTS "${CNPY_INCLUDE_DIR}/cnpy.h")
    message(STATUS "cnpy not found in ${CNPY_PATH}")
    message(STATUS "  build: bash scripts/third_party_builder.sh ${THIRD_PARTY_PLATFORM_TAG} --libs cnpy")
    return()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/FindZlibCustom.cmake")

if(NOT ZLIB_FOUND OR NOT TARGET ZLIB::ZLIB)
    message(STATUS "cnpy skipped: ZLIB not available")
    return()
endif()

if(NOT TARGET cnpy::cnpy)
    add_library(cnpy::cnpy STATIC IMPORTED)
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
message(STATUS "  target  : cnpy::cnpy")
message(STATUS "==============================================================================")
