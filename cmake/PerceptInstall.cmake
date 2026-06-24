include(GNUInstallDirs)

install(TARGETS root_sdk_percept
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY include/deploy_percept
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# apps/ 数据文件（源码唯一来源；示例与测试共用）
install(FILES
    apps/yolov5_detect_rknn/bus.jpg
    apps/yolov5_detect_rknn/coco_80_labels_list.txt
    apps/yolov5_detect_rknn/yolov5_detect_result_model_outputs.npz
    DESTINATION share/percept/apps/yolov5_detect_rknn)

install(FILES
    apps/yolov5_seg_rknn/bus.jpg
    apps/yolov5_seg_rknn/coco_80_labels_list.txt
    apps/yolov5_seg_rknn/yolov5_seg_result_model_outputs.npz
    apps/yolov5_seg_rknn/yolov5_seg_result_mask.bin
    DESTINATION share/percept/apps/yolov5_seg_rknn)

install(FILES
    apps/yolov8_seg_rknn/bus.jpg
    apps/yolov8_seg_rknn/coco_80_labels_list.txt
    apps/yolov8_seg_rknn/yolov8_seg_result_model_outputs.npz
    apps/yolov8_seg_rknn/yolov8_seg_result_mask.bin
    DESTINATION share/percept/apps/yolov8_seg_rknn)

install(FILES
    apps/yolov8_pose_rknn/bus.jpg
    DESTINATION share/percept/apps/yolov8_pose_rknn)
