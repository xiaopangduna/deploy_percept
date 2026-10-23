cmake_minimum_required(VERSION 3.10)

project(yolov8_detect_dump_sigmastar LANGUAGES CXX)

add_executable(yolov8_detect_dump_sigmastar main.cpp)

target_link_libraries(yolov8_detect_dump_sigmastar PRIVATE
    deploy_percept_core
    deploy_percept_utils
    nlohmann_json::nlohmann_json)

include(GNUInstallDirs)

set_target_properties(yolov8_detect_dump_sigmastar PROPERTIES
    INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")

install(TARGETS yolov8_detect_dump_sigmastar
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
