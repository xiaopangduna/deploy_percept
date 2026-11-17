cmake_minimum_required(VERSION 3.10)

project(classifier_dump_sigmastar LANGUAGES CXX)

add_executable(classifier_dump_sigmastar main.cpp)

target_link_libraries(classifier_dump_sigmastar PRIVATE
    deploy_percept_core
    deploy_percept_utils
    nlohmann_json::nlohmann_json)

include(GNUInstallDirs)

# cnpy 默认 SHARED：install 树为 bin/ + lib/libcnpy.so，需 $ORIGIN/../lib
set_target_properties(classifier_dump_sigmastar PROPERTIES
    INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")

install(TARGETS classifier_dump_sigmastar
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
