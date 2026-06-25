#include "deploy_percept/engine/EngineFactory.hpp"
#include "deploy_percept/engine/RknnEngine.hpp"
#ifdef AWNN_FOUND
#include "deploy_percept/engine/AwnnEngine.hpp"
#endif


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
#ifdef AWNN_FOUND
            case EngineType::AWNN:
                return std::make_unique<AwnnEngine>(AwnnEngine::Params());
#endif
            default:
                return nullptr;
            }
        }

    } // namespace engine
} // namespace deploy_percept