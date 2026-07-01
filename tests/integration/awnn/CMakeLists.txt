add_percept_awnn_integration_test(
    NAME test_yolov5_detect_awnn
    SOURCES test_yolov5_detect_awnn.cpp
    LINK_LIBS deploy_percept_core deploy_percept_utils test_utils
    USE_CUSTOM_MAIN)

add_percept_awnn_integration_test(
    NAME test_yolov8_detect_awnn
    SOURCES test_yolov8_detect_awnn.cpp
    LINK_LIBS deploy_percept_core deploy_percept_utils test_utils
    USE_CUSTOM_MAIN)
