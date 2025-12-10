# rknn runtime
set(TARGET_SOC "rk3588")

# for rknpu2
if (TARGET_SOC STREQUAL "rk3588" OR TARGET_SOC STREQUAL "rk3576" OR TARGET_SOC STREQUAL "rk356x" OR TARGET_SOC STREQUAL "rv1106" OR TARGET_SOC STREQUAL "rv1103" OR TARGET_SOC STREQUAL "rv1126b")
    set(RKNN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rknpu2)
    if (TARGET_SOC STREQUAL "rk3588" OR TARGET_SOC STREQUAL "rk356x" OR TARGET_SOC STREQUAL "rk3576")
        set(LIBRKNNRT ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}/librknnrt.so)
    endif()
    if (TARGET_SOC STREQUAL "rv1126b")
        set(LIBRKNNRT ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}/librknnrt.so)
    endif()
    if (TARGET_SOC STREQUAL "rv1106" OR TARGET_SOC STREQUAL "rv1103")
        set(LIBRKNNRT ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/armhf-uclibc/librknnmrt.so)
    endif()
    # 使用生成器表达式避免相对路径问题
    set(LIBRKNNRT_INCLUDES ${RKNN_PATH}/include)
endif()

# for rknpu1
if(TARGET_SOC STREQUAL "rk1808" OR TARGET_SOC STREQUAL "rv1109" OR TARGET_SOC STREQUAL "rv1126")
    set(RKNN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rknpu1)
    set(LIBRKNNRT ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}/librknn_api.so)
    # 使用生成器表达式避免相对路径问题
    set(LIBRKNNRT_INCLUDES ${RKNN_PATH}/include)
endif()

set(LIBRKNNRT ${LIBRKNNRT})

message(STATUS "==============================================================================")
if(EXISTS ${LIBRKNNRT})
    message(STATUS "RKNN runtime found successfully")
    message(STATUS "Target SoC: ${TARGET_SOC}")
    message(STATUS "RKNN path: ${RKNN_PATH}")
    message(STATUS "RKNN library: ${LIBRKNNRT}")
    message(STATUS "RKNN includes: ${LIBRKNNRT_INCLUDES}")
else()
    message(WARNING "RKNN runtime not found or not exists")
    message(WARNING "Target SoC: ${TARGET_SOC}")
    message(WARNING "Expected library path: ${LIBRKNNRT}")
endif()
message(STATUS "==============================================================================")

# install(PROGRAMS ${LIBRKNNRT} DESTINATION lib)

