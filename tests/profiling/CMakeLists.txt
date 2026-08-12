list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ProfilingHelpers)

add_percept_profiling_tool(
    NAME percept_monitor
    SOURCES percept_monitor.cpp
)

if(INSTALL_PROFILING)
    install(
        PROGRAMS "${CMAKE_SOURCE_DIR}/scripts/profile_run.sh"
        DESTINATION scripts)
endif()

