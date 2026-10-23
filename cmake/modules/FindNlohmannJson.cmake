# FindNlohmannJson.cmake - nlohmann/json（header-only，非致命）
#
# find_package(nlohmann_json) 设置 nlohmann_json_FOUND、nlohmann_json::nlohmann_json。

list(INSERT CMAKE_PREFIX_PATH 0 "${THIRD_PARTY_PLATFORM_DIR}/nlohmann_json")

find_package(nlohmann_json QUIET)

if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found under ${THIRD_PARTY_PLATFORM_DIR}/nlohmann_json")
    message(STATUS "  build: bash scripts/third_party_builder.sh ${THIRD_PARTY_PLATFORM_TAG} --libs nlohmann_json")
    return()
endif()

message(STATUS "==============================================================================")
message(STATUS "nlohmann_json found successfully")
message(STATUS "  version : ${nlohmann_json_VERSION}")
message(STATUS "  config  : ${nlohmann_json_CONFIG}")
message(STATUS "==============================================================================")
