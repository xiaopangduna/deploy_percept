#include "deploy_percept/engine/EngineFactory.hpp"
#include "deploy_percept/engine/RknnEngine.hpp"

namespace deploy_percept
{
    namespace engine
    {

        std::unique_ptr<BaseEngine> EngineFactory::create(EngineType type)
        {
            switch (type)
            {
            case EngineType::RKNN:
                return std::make_unique<RknnEngine>(RknnEngine::Params());
            default:
                return nullptr;
            }
        }

    } // namespace post_process
} // namespace deploy_percept