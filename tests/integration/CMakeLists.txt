# 子目录按引擎划分；各子目录自行用 *_FOUND 决定是否编译
if(AWNN_FOUND)
    add_subdirectory(awnn)
else()
    message(STATUS "integration/awnn: AWNN 未找到，跳过 AWNN 集成测试")
endif()
