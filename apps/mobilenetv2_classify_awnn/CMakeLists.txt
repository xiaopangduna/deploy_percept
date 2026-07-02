cmake_minimum_required(VERSION 3.10)

project(mobilenetv2_classify_awnn LANGUAGES CXX)

add_executable(mobilenetv2_classify_awnn main.cpp)

target_link_libraries(mobilenetv2_classify_awnn PRIVATE
    deploy_percept_core
    deploy_percept_utils)

install(TARGETS mobilenetv2_classify_awnn RUNTIME DESTINATION bin)

set(_mobilenetv2_app_dir "${CMAKE_CURRENT_SOURCE_DIR}")
if(EXISTS "${_mobilenetv2_app_dir}/mobilenetv2-12_pcq_a733.nb")
    install(FILES "${_mobilenetv2_app_dir}/mobilenetv2-12_pcq_a733.nb"
        DESTINATION share/percept/apps/mobilenetv2_classify_awnn
        RENAME mobilenetv2.nb)
elseif(EXISTS "${_mobilenetv2_app_dir}/mobilenetv2.nb")
    install(FILES "${_mobilenetv2_app_dir}/mobilenetv2.nb"
        DESTINATION share/percept/apps/mobilenetv2_classify_awnn)
else()
    message(WARNING "mobilenetv2_classify_awnn: model .nb not found under ${_mobilenetv2_app_dir}")
endif()

if(EXISTS "${_mobilenetv2_app_dir}/cat.jpg")
    install(FILES "${_mobilenetv2_app_dir}/cat.jpg"
        DESTINATION share/percept/apps/mobilenetv2_classify_awnn)
else()
    message(WARNING "mobilenetv2_classify_awnn: cat.jpg not found under ${_mobilenetv2_app_dir}")
endif()

set_target_properties(mobilenetv2_classify_awnn PROPERTIES
    BUILD_RPATH "${AWNN_VIPLITE_LIB_DIR}"
    INSTALL_RPATH "$ORIGIN/../lib")
