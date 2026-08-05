cmake_minimum_required(VERSION 3.10)

project(yolov8_detect_sigmastar LANGUAGES CXX)

add_executable(yolov8_detect_sigmastar main.cpp)

target_link_libraries(yolov8_detect_sigmastar PRIVATE
    deploy_percept_core
    deploy_percept_utils)

include(GNUInstallDirs)

# cnpy 默认 SHARED：install 树为 bin/ + lib/libcnpy.so，需 $ORIGIN/../lib
set_target_properties(yolov8_detect_sigmastar PROPERTIES
    INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")

install(TARGETS yolov8_detect_sigmastar
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
