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

# 创建库目标
add_library(deploy_percept STATIC
    src/calculator/Calculator.cpp
    src/utils/logger_factory.cpp
    src/post_process/YoloBasePostProcess.cpp
    src/post_process/YoloV5DetectPostProcess.cpp
    src/post_process/PostProcessFactory.cpp
)

# 链接OpenCV库到我们的库
target_link_libraries(deploy_percept PUBLIC ${OpenCV_LIBS} spdlog::spdlog ${LIBRKNNRT})

# 设置库的包含目录
target_include_directories(deploy_percept PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${LIBRKNNRT_INCLUDES}>
)

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