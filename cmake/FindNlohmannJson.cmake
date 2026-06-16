# FindNlohmannJson.cmake - nlohmann/json (header-only)

set(_NLOHMANN_JSON_INCLUDE_DIR "${THIRD_PARTY_DIR}/nlohmann/include")

if(NOT EXISTS "${_NLOHMANN_JSON_INCLUDE_DIR}/nlohmann/json.hpp")
    message(FATAL_ERROR
        "nlohmann/json not found:\n"
        "  ${_NLOHMANN_JSON_INCLUDE_DIR}/nlohmann/json.hpp")
endif()

if(NOT TARGET nlohmann_json::nlohmann_json)
    add_library(nlohmann_json INTERFACE)
    add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)
    target_include_directories(nlohmann_json INTERFACE "${_NLOHMANN_JSON_INCLUDE_DIR}")
endif()

set(NLOHMANN_JSON_FOUND TRUE)
set(NLOHMANN_JSON_INCLUDE_DIR "${_NLOHMANN_JSON_INCLUDE_DIR}")

message(STATUS "nlohmann/json found: ${NLOHMANN_JSON_INCLUDE_DIR}")
