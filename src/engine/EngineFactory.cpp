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
            default:
                return nullptr;
            }
        }

#ifdef AWNN_FOUND
        std::unique_ptr<AwnnEngine> EngineFactory::createAwnn(const AwnnEngine::Param &param)
        {
            return std::make_unique<AwnnEngine>(param);
        }
#endif

    } // namespace engine
} // namespace deploy_percept
