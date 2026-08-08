add_percept_sigmastar_integration_test(
    NAME test_yolov8_detect_sigmastar
    SOURCES test_yolov8_detect_sigmastar.cpp
    LINK_LIBS deploy_percept_core deploy_percept_utils)

add_percept_sigmastar_integration_test(
    NAME test_mobilenetv2_classify_sigmastar
    SOURCES test_mobilenetv2_classify_sigmastar.cpp
    LINK_LIBS deploy_percept_core deploy_percept_utils)
