# FindCnpyCustom.cmake - 自定义Cnpy库查找模块

# 设置cnpy路径，基于当前架构
set(CNPY_PATH ${THIRD_PARTY_DIR}/cnpy/${CMAKE_SYSTEM_PROCESSOR})

# 直接指定cnpy的头文件和库路径
set(CNPY_INCLUDE_DIRS ${CNPY_PATH}/include)
set(CNPY_LIBRARY ${CNPY_PATH}/lib/libcnpy.a)

# 检查头文件和库文件是否存在
if(EXISTS ${CNPY_LIBRARY} AND EXISTS ${CNPY_INCLUDE_DIRS}/cnpy.h)
    set(CNPY_FOUND TRUE)
    set(CNPY_LIBRARIES ${CNPY_LIBRARY})
    message(STATUS "CNPY found: ${CNPY_PATH}")
else()
    set(CNPY_FOUND FALSE)
    message(WARNING "CNPY not found, expected path: ${CNPY_PATH}")
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