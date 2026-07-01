# 示例应用：由平台 Find 结果（*_FOUND）控制，一平台一引擎
# 禁用某平台引擎：在 cmake/FindThirdParty.cmake 中去掉对应 include

# TODO: RKNN（瑞芯微）
# if(RKNN_FOUND)
#     add_subdirectory(yolov5_detect_rknn)
#     add_subdirectory(yolov5_seg_rknn)
#     add_subdirectory(yolov8_seg_rknn)
#     add_subdirectory(yolov8_pose_rknn)
# endif()

if(AWNN_FOUND)
    add_subdirectory(yolov5_detect_awnn)
    add_subdirectory(yolov8_detect_awnn)
else()
    message(STATUS "apps: AWNN 未找到，跳过 AWNN 示例（检查 third_party/.../awnn/viplite-tina）")
endif()
