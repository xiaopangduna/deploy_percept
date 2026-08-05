cmake_minimum_required(VERSION 3.10)

project(yolov8_detect_ipu LANGUAGES CXX)

add_executable(yolov8_detect_ipu
    main.cpp
    cmdline.cpp
    file.cpp)

target_include_directories(yolov8_detect_ipu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS})

target_link_libraries(yolov8_detect_ipu PRIVATE
    sigmastar::mi
    ${OpenCV_LIBS})

include(GNUInstallDirs)

install(TARGETS yolov8_detect_ipu
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
