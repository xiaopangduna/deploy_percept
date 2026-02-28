cmake_minimum_required(VERSION 3.10)

project(yolov8_seg_rknn)

add_executable(yolov8_seg_rknn main.cpp)

# 链接deploy_percept库和cnpy库
target_link_libraries(yolov8_seg_rknn PRIVATE deploy_percept ${CNPY_LIBRARIES})

# 添加cnpy头文件目录
target_include_directories(yolov8_seg_rknn PRIVATE ${CNPY_INCLUDE_DIRS})

# 安装可执行文件
install(TARGETS yolov8_seg_rknn
    RUNTIME DESTINATION bin
)