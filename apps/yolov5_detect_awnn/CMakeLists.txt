cmake_minimum_required(VERSION 3.10)

project(yolov5_detect_awnn LANGUAGES CXX)

add_executable(yolov5_detect_awnn main.cpp yolov5_fp32_post.cpp)

target_link_libraries(yolov5_detect_awnn PRIVATE
    deploy_percept
    AWNN::VIPLite
    pthread
    ${OpenCV_LIBS})

if(OpenCV_INCLUDE_DIRS)
    target_include_directories(yolov5_detect_awnn PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()

install(TARGETS yolov5_detect_awnn RUNTIME DESTINATION bin)

set_target_properties(yolov5_detect_awnn PROPERTIES
    BUILD_RPATH "${AWNN_VIPLITE_LIB_DIR}"
    INSTALL_RPATH "$ORIGIN/../lib")
