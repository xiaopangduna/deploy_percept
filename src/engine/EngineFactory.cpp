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
#if RKNN_FOUND
            case EngineType::RKNN:
                return std::make_unique<RknnEngine>(RknnEngine::Params());
#endif
            default:
                return nullptr;
            }
        }

    } // namespace engine
} // namespace deploy_percept