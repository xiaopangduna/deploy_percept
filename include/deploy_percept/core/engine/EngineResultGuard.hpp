#pragma once

#include "deploy_percept/core/engine/BaseEngine.hpp"

namespace deploy_percept
{
    namespace engine
    {

        /** run 成功后构造，析构时对 engine 调用 releaseResult() */
        class EngineResultGuard
        {
        public:
            explicit EngineResultGuard(BaseEngine &engine) : engine_(engine) {}

            ~EngineResultGuard() { engine_.releaseResult(); }

            const BaseEngine::Result &getEngineResult() const
            {
                return engine_.getResult();
            }

            EngineResultGuard(const EngineResultGuard &) = delete;
            EngineResultGuard &operator=(const EngineResultGuard &) = delete;

        private:
            BaseEngine &engine_;
        };

    } // namespace engine
} // namespace deploy_percept
