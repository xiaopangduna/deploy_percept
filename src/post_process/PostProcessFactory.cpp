#include "deploy_percept/post_process/PostProcessFactory.hpp"
#include "deploy_percept/post_process/YoloV5DetectPostProcess.hpp"

namespace deploy_percept {
namespace post_process {

std::unique_ptr<BasePostProcess> PostProcessFactory::create(PostProcessType type) {
    switch (type) {
        case PostProcessType::YOLOV5:
            return std::make_unique<YoloV5DetectPostProcess>();
        default:
            return nullptr;
    }
}

} // namespace post_process
} // namespace deploy_percept