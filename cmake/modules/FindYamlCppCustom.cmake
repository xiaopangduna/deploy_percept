# FindYamlCppCustom.cmake - 自定义 yaml-cpp 库查找模块（非致命）
#
# third_party 为 shared 时（SHARED IMPORTED），自动 install 运行时 .so。

list(INSERT CMAKE_PREFIX_PATH 0 "${THIRD_PARTY_PLATFORM_DIR}/yaml-cpp")

find_package(yaml-cpp QUIET)

if(NOT yaml-cpp_FOUND)
    message(STATUS "yaml-cpp not found under ${THIRD_PARTY_PLATFORM_DIR}/yaml-cpp")
    message(STATUS "  build: bash scripts/third_party_builder.sh ${THIRD_PARTY_PLATFORM_TAG} --libs yaml-cpp")
    return()
endif()

if(TARGET yaml-cpp::yaml-cpp)
    include(${CMAKE_CURRENT_LIST_DIR}/PerceptThirdPartyInstall.cmake)
    percept_install_imported_shared_runtime(yaml-cpp::yaml-cpp)
endif()

message(STATUS "==============================================================================")
message(STATUS "yaml-cpp found successfully")
message(STATUS "  version : ${yaml-cpp_VERSION}")
message(STATUS "  config  : ${yaml-cpp_CONFIG}")
message(STATUS "==============================================================================")
