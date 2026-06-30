cmake_minimum_required(VERSION 3.10)

project(yolov8_detect_awnn LANGUAGES CXX)

add_executable(yolov8_detect_awnn main.cpp)

target_link_libraries(yolov8_detect_awnn PRIVATE
    deploy_percept
    AWNN::VIPLite
    pthread)

install(TARGETS yolov8_detect_awnn RUNTIME DESTINATION bin)

set_target_properties(yolov8_detect_awnn PROPERTIES
    BUILD_RPATH "${AWNN_VIPLITE_LIB_DIR}"
    INSTALL_RPATH "$ORIGIN/../lib")
