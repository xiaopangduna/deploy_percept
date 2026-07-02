# 库分层：common（头文件契约）→ core（推理+后处理）→ utils（OpenCV 等辅助）
add_subdirectory(common)
add_subdirectory(core)
add_subdirectory(utils)
