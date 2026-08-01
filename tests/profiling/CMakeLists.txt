list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ProfilingHelpers)

add_percept_profiling_tool(
    NAME percept_monitor
    SOURCES percept_monitor.cpp
)
