add_percept_test(
    NAME test_YoloV5DetectPostProcess
    TIER unit
    SOURCES post_process/test_YoloV5DetectPostProcess.cpp
    LINK_LIBS deploy_percept test_utils
    USE_CUSTOM_MAIN)

add_percept_test(
    NAME test_YoloV5SegPostProcess
    TIER unit
    SOURCES post_process/test_YoloV5SegPostProcess.cpp
    LINK_LIBS deploy_percept test_utils
    USE_CUSTOM_MAIN)

add_percept_test(
    NAME test_YoloV8SegPostProcess
    TIER unit
    SOURCES post_process/test_YoloV8SegPostProcess.cpp
    LINK_LIBS deploy_percept test_utils
    USE_CUSTOM_MAIN)
