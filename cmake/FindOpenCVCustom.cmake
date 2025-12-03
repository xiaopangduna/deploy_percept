# FindOpenCVCustom.cmake - 自定义OpenCV库查找模块

#.rst:
# FindOpenCVCustom
# ----------------
#
# 自定义导入和配置OpenCV库
#
# 该模块会查找OpenCV库，并设置适当的变量和目标

# 查找并配置OpenCV

set(OpenCV_ROOT_DIR ${THIRD_PARTY_DIR}/third_party/opencv/ ${THIRD_PARTY_PLATFORM})
# 使用find_package查找OpenCV（Config模式）
find_package(OpenCV REQUIRED)