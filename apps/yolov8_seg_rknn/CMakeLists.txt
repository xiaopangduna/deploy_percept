cmake_minimum_required(VERSION 3.10)

project(yolov8_seg_rknn)

add_executable(yolov8_seg_rknn main.cpp)

target_link_libraries(yolov8_seg_rknn PRIVATE deploy_percept)

# 安装可执行文件
install(TARGETS yolov8_seg_rknn
    RUNTIME DESTINATION bin
)