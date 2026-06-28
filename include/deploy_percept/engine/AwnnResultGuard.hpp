#pragma once

#ifdef AWNN_FOUND

#include "deploy_percept/engine/AwnnEngine.hpp"

namespace deploy_percept
{
    namespace engine
    {

        /** run 成功后构造，析构时对 AwnnEngine 调用 releaseResult() */
        class AwnnResultGuard
        {
        public:
            explicit AwnnResultGuard(AwnnEngine &engine) : engine_(engine) {}
            ~AwnnResultGuard() { engine_.releaseResult(); }

            const std::vector<post_process::TensorView> &views() const
            {
                return engine_.getResult().outputs;
            }
            bool empty() const { return !engine_.getResult().ready; }

            AwnnResultGuard(const AwnnResultGuard &) = delete;
            AwnnResultGuard &operator=(const AwnnResultGuard &) = delete;

        private:
            AwnnEngine &engine_;
        };

    } // namespace engine
} // namespace deploy_percept

#endif
