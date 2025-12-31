#pragma once

#include "BaseEngine.hpp"
#include <memory>

namespace deploy_percept {
namespace engine {

class EngineFactory {
public:
    enum class EngineType {
        RKNN,
        // 其他类型可以在这里添加
    };
    
    static std::unique_ptr<BaseEngine> create(EngineType type);
};

} // namespace engine
} // namespace deploy_percept