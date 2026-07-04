add_percept_test(
    NAME test_YoloV5DetectPostProcessRknn
    TIER unit
    SOURCES post_process/test_YoloV5DetectPostProcessRknn.cpp
    LINK_LIBS deploy_percept_core deploy_percept_utils test_utils
    USE_CUSTOM_MAIN)

add_percept_test(
    NAME test_YoloV5SegPostProcess
    TIER unit
    SOURCES post_process/test_YoloV5SegPostProcess.cpp
    LINK_LIBS deploy_percept_core deploy_percept_utils test_utils
    USE_CUSTOM_MAIN)

add_percept_test(
    NAME test_YoloV8SegPostProcess
    TIER unit
    SOURCES post_process/test_YoloV8SegPostProcess.cpp
    LINK_LIBS deploy_percept_core deploy_percept_utils test_utils
    USE_CUSTOM_MAIN)

add_percept_test(
    NAME test_ClassificationPostProcessAwnn
    TIER unit
    SOURCES post_process/test_ClassificationPostProcessAwnn.cpp
    LINK_LIBS deploy_percept_core deploy_percept_utils test_utils)
