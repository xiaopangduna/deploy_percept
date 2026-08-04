set(CORE_SOURCES
    engine/BaseEngine.cpp
    post_process/YoloBasePostProcess.cpp
    post_process/YoloDetectPostProcess.cpp
    post_process/YoloV5DetectPostProcessRknn.cpp
    post_process/YoloV5DetectPostProcessAwnn.cpp
    post_process/YoloV8DetectPostProcessAwnn.cpp
    post_process/YoloV5SegPostProcessRknn.cpp
    post_process/YoloV8PosePostProcessRknn.cpp
    post_process/YoloV8SegPostProcessRknn.cpp
    post_process/ClassificationPostProcessAwnn.cpp
)

add_library(deploy_percept_core STATIC ${CORE_SOURCES})

target_link_libraries(deploy_percept_core PUBLIC deploy_percept_common)

# 平台 NPU 引擎：由 FindThirdParty 按 platform tag 探测，*_FOUND 即编入核心库
# TODO: Sigmastar SSD2355 (CoMake Pi D1)
# if(SIGMASTAR_FOUND)
#     target_sources(deploy_percept_core PRIVATE
#         engine/SigmastarIpuEngine.cpp
#     )
#     target_link_libraries(deploy_percept_core PUBLIC sigmastar::mi)
#     target_compile_definitions(deploy_percept_core PUBLIC SIGMASTAR_FOUND=1)
# endif()

# TODO: RKNN（瑞芯微）
# if(RKNN_FOUND)
#     target_sources(deploy_percept_core PRIVATE
#         engine/RknnEngine.cpp
#     )
#     target_link_libraries(deploy_percept_core PUBLIC RKNN::runtime)
#     target_compile_definitions(deploy_percept_core PUBLIC RKNN_FOUND=1)
# endif()

if(AWNN_FOUND)
    target_sources(deploy_percept_core PRIVATE
        engine/AwnnEngine.cpp
    )
    target_link_libraries(deploy_percept_core PUBLIC AWNN::VIPLite pthread)
    target_compile_definitions(deploy_percept_core PUBLIC AWNN_FOUND=1)
endif()

include(GNUInstallDirs)

install(TARGETS deploy_percept_core
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
