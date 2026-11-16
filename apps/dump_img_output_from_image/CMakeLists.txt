cmake_minimum_required(VERSION 3.10)

project(dump_img_output_from_image LANGUAGES CXX)

add_executable(dump_img_output_from_image main.cpp)

target_link_libraries(dump_img_output_from_image PRIVATE
    deploy_percept_core
    deploy_percept_utils
    nlohmann_json::nlohmann_json)

include(GNUInstallDirs)

# cnpy 默认 SHARED：install 树为 bin/ + lib/libcnpy.so，需 $ORIGIN/../lib
set_target_properties(dump_img_output_from_image PROPERTIES
    INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")

install(TARGETS dump_img_output_from_image
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
