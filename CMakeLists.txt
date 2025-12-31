cmake_minimum_required(VERSION 3.10)

# 定义项目名称
project(deploy_percept VERSION 0.1.0 LANGUAGES CXX)

# 指定C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 将cmake目录添加到模块查找路径
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# 包含第三方库查找模块
include(FindThirdParty)

# 添加包含目录
include_directories(include)

# 添加子目录

# 定义基础源文件列表（不包含RGA相关文件）
set(DEPLOY_PERCEPT_SOURCES
    src/pre_process/image_pre_process.cpp
    src/engine/RknnEngine.cpp
    src/engine/EngineFactory.cpp
    src/engine/BaseEngine.cpp
    src/post_process/YoloBasePostProcess.cpp
    src/post_process/YoloV5DetectPostProcess.cpp
    src/post_process/PostProcessFactory.cpp
    src/utils/logger_factory.cpp
)

# 创建库目标
add_library(deploy_percept STATIC ${DEPLOY_PERCEPT_SOURCES})


# 链接OpenCV库、spdlog、RKNN、cnpy和zlib库到我们的库

target_link_libraries(deploy_percept PUBLIC ${OpenCV_LIBS} spdlog::spdlog ${LIBRKNNRT} ${CNPY_LIBRARIES} ${YAML_CPP_LIBRARIES} z )


# 设置库的包含目录
target_include_directories(deploy_percept PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${LIBRKNNRT_INCLUDES}>
    $<BUILD_INTERFACE:${CNPY_INCLUDE_DIRS}>
    ${YAML_CPP_INCLUDE_DIR}
)

# RGA
if(RGA_FOUND)
    target_sources(deploy_percept PRIVATE src/pre_process/image_pre_process_rga.cpp)
    target_link_libraries(deploy_percept PUBLIC ${LIBRGA})
    target_include_directories(deploy_percept PUBLIC
    $<BUILD_INTERFACE:${LIBRGA_INCLUDES}>
)    
    target_compile_definitions(deploy_percept PUBLIC RGA_FOUND=1)
endif()


# 安装规则
install(TARGETS deploy_percept
    EXPORT deploy_perceptTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/
    DESTINATION include
)

install(EXPORT deploy_perceptTargets
    FILE deploy_perceptConfig.cmake
    NAMESPACE deploy_percept::
    DESTINATION lib/cmake/deploy_percept
)

# 添加子目录
add_subdirectory(apps)
# 启用测试
enable_testing()
add_subdirectory(tests)