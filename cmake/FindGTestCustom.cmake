# FindGTestCustom.cmake - 自定义Google Test库查找模块

#.rst:
# FindGTestCustom
# ---------------
#
# 自定义导入和配置Google Test库
#
# 该模块会查找Google Test库，并设置适当的变量和目标

# 查找并配置Google Test

set(GTEST_ROOT_DIR ${THIRD_PARTY_DIR}/third_party/gtest/ ${THIRD_PARTY_PLATFORM})
# 使用find_package查找GTest（Config模式）
find_package(GTest REQUIRED)