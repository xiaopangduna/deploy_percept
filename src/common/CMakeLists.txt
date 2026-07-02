add_library(deploy_percept_common INTERFACE)

target_include_directories(deploy_percept_common INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
