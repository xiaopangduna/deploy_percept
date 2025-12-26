#.rst:
# FindCnpy
# --------
#
# Find cnpy library
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ``CNPY_FOUND``
#     True if cnpy library found
# ``CNPY_VERSION``
#     The version of cnpy library
# ``CNPY_INCLUDE_DIRS``
#     The include directories for cnpy library
# ``CNPY_LIBRARIES``
#     The libraries needed to use cnpy
#
# Hints
# ^^^^^
# A user may set ``CNPY_ROOT`` to an installation root to tell this module where to look.

unset(CNPY_FOUND)

# Set default values for CNPY variables
set(CNPY_ROOT "${THIRD_PARTY_DIR}/cnpy/${CMAKE_SYSTEM_PROCESSOR}" CACHE PATH "CNPY root directory")

set(CNPY_INCLUDE_DIR "${CNPY_ROOT}/include" CACHE PATH "CNPY include directory")
set(CNPY_LIBRARY_DIR "${CNPY_ROOT}/lib" CACHE PATH "CNPY library directory")

# Find header
find_path(CNPY_INCLUDE_DIR
    NAMES cnpy.h
    PATHS ${CNPY_ROOT}/include
    PATH_SUFFIXES cnpy
    NO_DEFAULT_PATH
)

# Find library
find_library(CNPY_LIBRARY
    NAMES cnpy
    PATHS ${CNPY_ROOT}/lib
    NO_DEFAULT_PATH
)

# Check if both header and library are found
if(CNPY_INCLUDE_DIR AND CNPY_LIBRARY)
    set(CNPY_FOUND TRUE)
    set(CNPY_INCLUDE_DIRS ${CNPY_INCLUDE_DIR})
    set(CNPY_LIBRARIES ${CNPY_LIBRARY})
    message(STATUS "Found CNPY: ${CNPY_ROOT}")
else()
    message(STATUS "CNPY not found, expected path: ${CNPY_ROOT}")
endif()

mark_as_advanced(CNPY_INCLUDE_DIR CNPY_LIBRARY)

# FindCnpyCustom.cmake - 自定义cnpy库查找模块

set(CNPY_PATH ${THIRD_PARTY_DIR}/cnpy/${CMAKE_SYSTEM_PROCESSOR})
set(CNPY_LIBRARY ${CNPY_PATH}/lib/libcnpy.a)
set(CNPY_INCLUDE_DIRS ${CNPY_PATH}/include)

# 检查库文件是否存在
if(EXISTS ${CNPY_LIBRARY})
    set(CNPY_LIBRARIES ${CNPY_LIBRARY})
    set(CNPY_FOUND TRUE)
else()
    set(CNPY_FOUND FALSE)
endif()

message(STATUS "==============================================================================")
if(CNPY_FOUND)
    message(STATUS "CNPY found successfully")
    message(STATUS "CNPY include dirs: ${CNPY_INCLUDE_DIRS}")
    message(STATUS "CNPY libraries: ${CNPY_LIBRARIES}")
else()
    message(FATAL_ERROR "CNPY not found")
endif()
message(STATUS "==============================================================================")
