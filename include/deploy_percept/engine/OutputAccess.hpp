#pragma once

#include "deploy_percept/engine/BaseEngine.hpp"

namespace deploy_percept
{
    namespace engine
    {

        /**
         * run 成功后构造，析构时调用 release_output_views()。
         * 适用于任意 BaseEngine 子类（AWNN Mapped / HostCopy、RKNN 等）。
         */
        class OutputAccess
        {
        public:
            explicit OutputAccess(BaseEngine &engine);
            ~OutputAccess();

            const std::vector<post_process::TensorView> &views() const { return views_; }
            bool empty() const { return views_.empty(); }

            OutputAccess(const OutputAccess &) = delete;
            OutputAccess &operator=(const OutputAccess &) = delete;

        private:
            BaseEngine &engine_;
            std::vector<post_process::TensorView> views_;
        };

    } // namespace engine
} // namespace deploy_percept
