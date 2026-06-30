if(NOT AWNN_FOUND)
    message(STATUS "integration/awnn: AWNN 未找到，跳过 AWNN 集成测试")
    return()
endif()

add_percept_awnn_integration_test(
    NAME test_yolov5_detect_awnn
    SOURCES test_yolov5_detect_awnn.cpp
    LINK_LIBS deploy_percept test_utils
    USE_CUSTOM_MAIN)

add_percept_awnn_integration_test(
    NAME test_yolov8_detect_awnn
    SOURCES test_yolov8_detect_awnn.cpp
    LINK_LIBS deploy_percept test_utils
    USE_CUSTOM_MAIN)
