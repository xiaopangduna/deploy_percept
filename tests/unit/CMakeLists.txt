add_percept_test(
    NAME test_YoloV5DetectPostProcessRknn
    TIER unit
    SOURCES post_process/test_YoloV5DetectPostProcessRknn.cpp
    LINK_LIBS deploy_percept_core deploy_percept_utils)

add_percept_test(
    NAME test_YoloV5SegPostProcessRknn
    TIER unit
    SOURCES post_process/test_YoloV5SegPostProcessRknn.cpp
    LINK_LIBS deploy_percept_core deploy_percept_utils)

add_percept_test(
    NAME test_YoloV8SegPostProcessRknn
    TIER unit
    SOURCES post_process/test_YoloV8SegPostProcessRknn.cpp
    LINK_LIBS deploy_percept_core deploy_percept_utils)

add_percept_test(
    NAME test_ClassificationPostProcessAwnn
    TIER unit
    SOURCES post_process/test_ClassificationPostProcessAwnn.cpp
    LINK_LIBS deploy_percept_core deploy_percept_utils)

add_percept_test(
    NAME test_ClassificationPostProcessSigmastar
    TIER unit
    SOURCES post_process/test_ClassificationPostProcessSigmastar.cpp
    LINK_LIBS deploy_percept_core)

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
