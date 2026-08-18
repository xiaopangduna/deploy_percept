cmake_minimum_required(VERSION 3.10)

project(yolo_cls_awnn LANGUAGES CXX)

add_executable(yolo_cls_awnn main.cpp)

target_link_libraries(yolo_cls_awnn PRIVATE
    deploy_percept_core
    deploy_percept_utils)

include(GNUInstallDirs)

set_target_properties(yolo_cls_awnn PROPERTIES
    BUILD_RPATH "${AWNN_VIPLITE_LIB_DIR}"
    INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")

install(TARGETS yolo_cls_awnn
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
