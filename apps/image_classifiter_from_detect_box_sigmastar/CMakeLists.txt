cmake_minimum_required(VERSION 3.10)

project(image_classifiter_from_detect_box_sigmastar LANGUAGES CXX)

add_executable(image_classifiter_from_detect_box_sigmastar main.cpp)

target_link_libraries(image_classifiter_from_detect_box_sigmastar PRIVATE
    deploy_percept_core
    deploy_percept_utils)

include(GNUInstallDirs)

# cnpy 默认 SHARED：install 树为 bin/ + lib/libcnpy.so，需 $ORIGIN/../lib
set_target_properties(image_classifiter_from_detect_box_sigmastar PROPERTIES
    INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")

install(TARGETS image_classifiter_from_detect_box_sigmastar
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
