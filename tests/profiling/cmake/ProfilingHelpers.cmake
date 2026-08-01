include(GNUInstallDirs)

# 源码在 tests/profiling/；install 后与 bin/、lib/、benchmarks/ 同级
set(PERCEPT_PROFILING_INSTALL_DIR "profiling")
# prefix/profiling/<exe> → prefix/lib
set(PERCEPT_PROFILING_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")

function(percept_install_profiling)
    if(NOT INSTALL_PROFILING)
        return()
    endif()
    if(NOT TARGET ${ARGV0})
        message(FATAL_ERROR "percept_install_profiling: unknown target '${ARGV0}'")
    endif()
    install(TARGETS ${ARGV0} RUNTIME DESTINATION ${PERCEPT_PROFILING_INSTALL_DIR})
    set_target_properties(${ARGV0} PROPERTIES
        INSTALL_RPATH "${PERCEPT_PROFILING_INSTALL_RPATH}")
endfunction()

# add_percept_profiling_tool - 无第三方依赖的旁路 profiling 工具
function(add_percept_profiling_tool)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES)
    cmake_parse_arguments(PARSED "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT PARSED_NAME)
        message(FATAL_ERROR "add_percept_profiling_tool: NAME is required")
    endif()
    if(NOT PARSED_SOURCES)
        message(FATAL_ERROR "add_percept_profiling_tool(${PARSED_NAME}): SOURCES is required")
    endif()

    add_executable(${PARSED_NAME} ${PARSED_SOURCES})
    target_compile_features(${PARSED_NAME} PRIVATE cxx_std_17)
    percept_install_profiling(${PARSED_NAME})
endfunction()
