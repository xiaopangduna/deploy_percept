cmake_minimum_required(VERSION 3.10)

project(yolov5_detect_rknn)

add_executable(yolov5_detect_rknn main.cpp)

target_link_libraries(yolov5_detect_rknn PRIVATE deploy_percept)

# 安装可执行文件
install(TARGETS yolov5_detect_rknn
    RUNTIME DESTINATION bin
)