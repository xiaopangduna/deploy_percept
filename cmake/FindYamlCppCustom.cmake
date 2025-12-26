# FindYamlCppCustom.cmake - 自定义yaml-cpp库查找模块

# 推荐方式：直接设置YAML_CPP_DIR变量指向配置文件目录
# set(YAML_CPP_DIR ${THIRD_PARTY_DIR}/yaml-cpp/${CMAKE_SYSTEM_PROCESSOR}/lib/cmake/yaml-cpp)
# 备选方式：将yaml-cpp路径添加到CMAKE_PREFIX_PATH中，让CMake自动查找
list(INSERT CMAKE_PREFIX_PATH 0 "${THIRD_PARTY_DIR}/yaml-cpp/${CMAKE_SYSTEM_PROCESSOR}")

find_package(yaml-cpp REQUIRED)

message(STATUS "==============================================================================")
if(yaml-cpp_FOUND)
    message(STATUS "yaml-cpp found successfully")
    message(STATUS "yaml-cpp version: ${yaml-cpp_VERSION}")
    message(STATUS "yaml-cpp libraries: ${YAML_CPP_LIBRARIES}")
    message(STATUS "yaml-cpp config file: ${yaml-cpp_CONFIG}")
else()
    message(FATAL_ERROR "yaml-cpp not found")
endif()
message(STATUS "==============================================================================")