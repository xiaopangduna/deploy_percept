#pragma once

#include "deploy_percept/core/engine/BaseEngine.hpp"

namespace deploy_percept
{
    namespace engine
    {

        // ==================================================================
        // EngineResultGuard — run 输出生命周期的 RAII 封装
        //
        // 在 engine.run() 成功之后构造；guard 存活期间可安全读取 tensor_views，
        // 析构时自动调用 engine.releaseResult()（unmap VIP buffer 等）。
        //
        // is_copy_host==true（默认）时 run() 已将输出拷至 host 并 unmap，Guard 可选。
        //
        // 典型用法（零拷贝）：
        //   engine.run(input, size);
        //   EngineResultGuard guard(engine);
        //   processor.run(guard.getEngineResult().tensor_views);
        // ==================================================================

        /**
         * 持有 BaseEngine 引用，析构时释放最近一次 run 的 mapped 输出。
         *
         * 不可拷贝；同一 engine 上勿嵌套多个 guard（后者析构会提前释放前者仍需要的视图）。
         */
        class EngineResultGuard
        {
        public:
            explicit EngineResultGuard(BaseEngine &engine) : engine_(engine) {}

            ~EngineResultGuard() { engine_.releaseResult(); }

            /** 只读访问 engine 当前 Result；guard 析构前有效 */
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
