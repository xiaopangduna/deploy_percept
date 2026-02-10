cmake_minimum_required(VERSION 3.10)

project(yolov5_seg_rknn)

add_executable(yolov5_seg_rknn main.cpp)

target_link_libraries(yolov5_seg_rknn PRIVATE deploy_percept)

# 安装可执行文件
install(TARGETS yolov5_detect_rknn
    RUNTIME DESTINATION bin
)