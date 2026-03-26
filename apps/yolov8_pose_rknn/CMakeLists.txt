cmake_minimum_required(VERSION 3.10)

project(yolov8_pose_rknn)

add_executable(yolov8_pose_rknn main.cpp)

target_link_libraries(yolov8_pose_rknn PRIVATE deploy_percept)

install(TARGETS yolov8_pose_rknn
    RUNTIME DESTINATION bin
)
