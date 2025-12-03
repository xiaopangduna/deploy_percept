cmake_minimum_required(VERSION 3.10)

# 定义项目名称
project(skeleton_cpp VERSION 0.1.0 LANGUAGES CXX)

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
add_library(skeleton_cpp STATIC
    src/calculator/Calculator.cpp
)

# 链接OpenCV库到我们的库
target_link_libraries(skeleton_cpp PRIVATE ${OpenCV_LIBS})

# 设置库的包含目录
target_include_directories(skeleton_cpp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# 安装规则
install(TARGETS skeleton_cpp
    EXPORT skeleton_cppTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/
    DESTINATION include
)

install(EXPORT skeleton_cppTargets
    FILE skeleton_cppConfig.cmake
    NAMESPACE skeleton_cpp::
    DESTINATION lib/cmake/skeleton_cpp
)

# 添加子目录
add_subdirectory(apps)
# 启用测试
enable_testing()
add_subdirectory(tests)