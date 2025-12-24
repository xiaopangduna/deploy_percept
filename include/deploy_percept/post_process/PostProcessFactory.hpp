#ifndef DEPLOY_PERCEPT_POST_PROCESS_POSTPROCESSFACTORY_HPP
#define DEPLOY_PERCEPT_POST_PROCESS_POSTPROCESSFACTORY_HPP

#include "BasePostProcess.hpp"
#include "YoloV5DetectPostProcess.hpp"
#include "YoloBasePostProcess.hpp"
#include <memory>

namespace deploy_percept {
namespace post_process {

class PostProcessFactory {
public:
    enum class PostProcessType {
        YOLOV5,
        // 其他类型可以在这里添加
    };
    
    static std::unique_ptr<BasePostProcess> create(PostProcessType type);
};

} // namespace post_process
} // namespace deploy_percept

#endif // DEPLOY_PERCEPT_POST_PROCESS_POSTPROCESSFACTORY_HPP