#pragma once

#include "BaseEngine.hpp"
#include <memory>

#ifdef AWNN_FOUND
#include "deploy_percept/engine/AwnnEngine.hpp"
#endif

namespace deploy_percept {
namespace engine {

class EngineFactory {
public:
    enum class EngineType {
        RKNN,
#ifdef AWNN_FOUND
        AWNN,
#endif
    };
    
    static std::unique_ptr<BaseEngine> create(EngineType type);
#ifdef AWNN_FOUND
    static std::unique_ptr<AwnnEngine> createAwnn(const AwnnEngine::Param &param);
#endif
};

} // namespace engine
} // namespace deploy_percept
