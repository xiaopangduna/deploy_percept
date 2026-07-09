# FindSpdlogCustom.cmake - 自定义 spdlog 库查找模块（非致命）
#
# find_package(spdlog) 会设置 spdlog_FOUND、spdlog::spdlog 等。
# third_party 为 shared 编译库时（SHARED IMPORTED），自动 install 运行时 .so。

list(INSERT CMAKE_PREFIX_PATH 0 "${THIRD_PARTY_PLATFORM_DIR}/spdlog")

find_package(spdlog QUIET)

if(NOT spdlog_FOUND)
    message(STATUS "spdlog not found under ${THIRD_PARTY_PLATFORM_DIR}/spdlog")
    message(STATUS "  build: bash scripts/third_party_builder.sh ${THIRD_PARTY_PLATFORM_TAG} --libs spdlog")
    return()
endif()

if(TARGET spdlog::spdlog)
    include(${CMAKE_CURRENT_LIST_DIR}/PerceptThirdPartyInstall.cmake)
    percept_install_imported_shared_runtime(spdlog::spdlog)
endif()

message(STATUS "==============================================================================")
message(STATUS "spdlog found successfully")
message(STATUS "  version : ${spdlog_VERSION}")
message(STATUS "  include : ${spdlog_INCLUDE_DIRS}")
message(STATUS "  config  : ${spdlog_CONFIG}")
message(STATUS "==============================================================================")
