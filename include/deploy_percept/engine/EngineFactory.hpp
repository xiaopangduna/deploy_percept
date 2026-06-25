#pragma once

#include "BaseEngine.hpp"
#include <memory>

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
};

} // namespace engine
} // namespace deploy_percept