if(NOT CNPY_FOUND OR NOT OpenCV_FOUND OR NOT spdlog_FOUND)
    message(WARNING
        "deploy_percept_utils skipped: need cnpy + OpenCV + spdlog"
        " (cnpy=${CNPY_FOUND} opencv=${OpenCV_FOUND} spdlog=${spdlog_FOUND})")
    return()
endif()

set(UTILS_SOURCES
    logger_factory.cpp
    io.cpp
    vis_draw.cpp
    npy.cpp
    fixture_path.cpp
    pre_process/image_pre_process.cpp
)

if(RGA_FOUND)
    list(APPEND UTILS_SOURCES pre_process/image_pre_process_rga.cpp)
endif()

add_library(deploy_percept_utils STATIC ${UTILS_SOURCES})

target_link_libraries(deploy_percept_utils PUBLIC deploy_percept_common)

target_include_directories(deploy_percept_utils PUBLIC
    $<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>)

target_link_libraries(deploy_percept_utils PUBLIC
    ${OpenCV_LIBS}
    spdlog::spdlog
    cnpy::cnpy)

if(RGA_FOUND)
    target_include_directories(deploy_percept_utils PUBLIC ${LIBRGA_INCLUDES})
    target_link_libraries(deploy_percept_utils PUBLIC ${LIBRGA})
endif()

include(GNUInstallDirs)

install(TARGETS deploy_percept_utils
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
