cmake_minimum_required(VERSION 3.10)

project(demo_app)

add_executable(demo_app main.cpp)

target_link_libraries(demo_app PRIVATE skeleton_cpp)

# 安装可执行文件
install(TARGETS demo_app
    RUNTIME DESTINATION bin
)