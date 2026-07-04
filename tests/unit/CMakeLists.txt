add_percept_test(
    NAME test_YoloV5DetectPostProcessRknn
    TIER unit
    SOURCES post_process/test_YoloV5DetectPostProcessRknn.cpp
    LINK_LIBS deploy_percept_core deploy_percept_utils)

add_percept_test(
    NAME test_YoloV5SegPostProcess
    TIER unit
    SOURCES post_process/test_YoloV5SegPostProcess.cpp
    LINK_LIBS deploy_percept_core deploy_percept_utils)

add_percept_test(
    NAME test_YoloV8SegPostProcess
    TIER unit
    SOURCES post_process/test_YoloV8SegPostProcess.cpp
    LINK_LIBS deploy_percept_core deploy_percept_utils)

add_percept_test(
    NAME test_ClassificationPostProcessAwnn
    TIER unit
    SOURCES post_process/test_ClassificationPostProcessAwnn.cpp
    LINK_LIBS deploy_percept_core deploy_percept_utils)

add_percept_test(
    NAME test_YoloV5DetectPostProcessAwnn
    TIER unit
    SOURCES post_process/test_YoloV5DetectPostProcessAwnn.cpp
    LINK_LIBS deploy_percept_core deploy_percept_utils)

add_percept_test(
    NAME test_YoloV8DetectPostProcessAwnn
    TIER unit
    SOURCES post_process/test_YoloV8DetectPostProcessAwnn.cpp
    LINK_LIBS deploy_percept_core deploy_percept_utils)
