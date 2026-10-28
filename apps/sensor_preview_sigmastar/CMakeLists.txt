cmake_minimum_required(VERSION 3.10)

project(sensor_preview_sigmastar LANGUAGES CXX)

add_executable(sensor_preview_sigmastar
    main.cpp
    detect_session.cpp)

target_link_libraries(sensor_preview_sigmastar PRIVATE
    sigmastar::sensor_pipeline
    deploy_percept_core)

include(GNUInstallDirs)

install(TARGETS sensor_preview_sigmastar
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
