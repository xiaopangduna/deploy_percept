cmake_minimum_required(VERSION 3.10)

project(mobilenetv2_classify_sigmastar LANGUAGES CXX)

add_executable(mobilenetv2_classify_sigmastar main.cpp)

target_link_libraries(mobilenetv2_classify_sigmastar PRIVATE
    deploy_percept_core
    deploy_percept_utils)

include(GNUInstallDirs)

# cnpy 默认 SHARED：install 树为 bin/ + lib/libcnpy.so，需 $ORIGIN/../lib
set_target_properties(mobilenetv2_classify_sigmastar PROPERTIES
    INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")

install(TARGETS mobilenetv2_classify_sigmastar
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
