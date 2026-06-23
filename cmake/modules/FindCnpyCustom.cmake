# FindCnpyCustom.cmake - 自定义 Cnpy 库查找模块

set(CNPY_PATH "${THIRD_PARTY_PLATFORM_DIR}/cnpy")
set(CNPY_INCLUDE_DIR "${CNPY_PATH}/include")
set(CNPY_LIBRARY "${CNPY_PATH}/lib/libcnpy.a")

if(NOT EXISTS "${CNPY_LIBRARY}" OR NOT EXISTS "${CNPY_INCLUDE_DIR}/cnpy.h")
    message(FATAL_ERROR
        "cnpy not found in ${CNPY_PATH}\n"
        "Build with:\n"
        "  bash scripts/third_party_builder.sh ${THIRD_PARTY_PLATFORM_TAG} --libs cnpy")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/FindZlibCustom.cmake")

if(NOT TARGET ZLIB::ZLIB)
    message(FATAL_ERROR "ZLIB::ZLIB target missing after FindZlibCustom.cmake")
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
