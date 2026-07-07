cmake_minimum_required(VERSION 3.10)

project(mobilenetv2_classify_awnn LANGUAGES CXX)

add_executable(mobilenetv2_classify_awnn main.cpp)

target_link_libraries(mobilenetv2_classify_awnn PRIVATE
    deploy_percept_core
    deploy_percept_utils)

include(GNUInstallDirs)

set_target_properties(mobilenetv2_classify_awnn PROPERTIES
    BUILD_RPATH "${AWNN_VIPLITE_LIB_DIR}"
    INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")

install(TARGETS mobilenetv2_classify_awnn
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
