cmake_minimum_required(VERSION 3.10)

# 定义项目名称
project(root_sdk_percept VERSION 0.1.0 LANGUAGES CXX)

option(ENABLE_TESTS "Enable unit tests" ON)
option(ENABLE_ROOT_SDK "Build root_sdk_percept library and RKNN apps" ON)

# 指定C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 工具链/平台变量见各 toolchain 文件；此处仅打印 project() 之后的工程配置
message(STATUS "==============================================================================")
message(STATUS "项目配置:")
message(STATUS "  CMAKE_BUILD_TYPE           = ${CMAKE_BUILD_TYPE}")
message(STATUS "  CMAKE_CXX_STANDARD         = ${CMAKE_CXX_STANDARD}")
message(STATUS "  CMAKE_CROSSCOMPILING       = ${CMAKE_CROSSCOMPILING}")
message(STATUS "  CMAKE_CXX_COMPILER_ID      = ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  CMAKE_CXX_COMPILER_VERSION = ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "==============================================================================")

# 将cmake目录添加到模块查找路径
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if(ENABLE_ROOT_SDK)
# 包含第三方库查找模块
include(FindThirdParty)

# 定义源文件列表
set(SOURCES_FILE
    src/utils/logger_factory.cpp
    # src/pre_process/image_pre_process.cpp
    # src/post_process/YoloBasePostProcess.cpp
    # src/post_process/YoloV8DetectPostProcess.cpp
)

# 创建库目标
add_library(root_sdk_percept STATIC ${SOURCES_FILE})

target_link_libraries(root_sdk_percept PUBLIC
    # ${OpenCV_LIBS}
    # spdlog::spdlog
    # yaml-cpp::yaml-cpp
)

# 设置库的包含目录
target_include_directories(root_sdk_percept PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Sigmastar MI SDK（armv7l SSC375 平台后端，条件链接）
if(SIGMASTAR_FOUND)
    target_sources(root_sdk_percept PRIVATE
        src/engine/SigmastarIpuEngine.cpp
    )
    target_link_libraries(root_sdk_percept PUBLIC sigmastar::mi)
    target_compile_definitions(root_sdk_percept PUBLIC SIGMASTAR_FOUND=1)
endif()

# # 安装规则（armv7l：不 EXPORT，MI SDK 为 INTERFACE 依赖）
# install(TARGETS root_sdk_percept
#     LIBRARY DESTINATION lib
#     ARCHIVE DESTINATION lib
#     RUNTIME DESTINATION bin
#     INCLUDES DESTINATION include
# )

# install(DIRECTORY include/
#     DESTINATION include
# )
endif()

# 添加apps目录
add_subdirectory(apps)

if(ENABLE_ROOT_SDK)
message(STATUS "==============================================================================")
message(STATUS "测试配置信息:")
message(STATUS "  - ENABLE_TESTS: ${ENABLE_TESTS}")
message(STATUS "  - GTest_FOUND: ${GTest_FOUND}")
message(STATUS "==============================================================================")
if(ENABLE_TESTS )
    if(GTest_FOUND)
        message(STATUS "测试已被启用，开始编译测试用例")
        enable_testing()
        add_subdirectory(tests)
    else()
        message(WARNING "测试已被启用，但未找到 Google Test，测试将被跳过")
    endif()
else()
    message(STATUS "测试已被禁用")
endif()
message(STATUS "==============================================================================")
else()
    message(STATUS "root_sdk_percept 已禁用，仅构建 apps/demo 等轻量目标")
endif()