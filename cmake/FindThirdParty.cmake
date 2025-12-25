# FindThirdParty.cmake - 第三方库路径设置模块

#.rst:
# 设置THIRD_PARTY_DIR目录为第三方库的路径

message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")


# 设置第三方库目录
set(THIRD_PARTY_DIR "${CMAKE_SOURCE_DIR}/third_party")

# 导入自定义的查找模块
include(FindOpenCVCustom)
include(FindSpdlogCustom)
include(FindRknn)

message(STATUS "已完成第三方库配置")