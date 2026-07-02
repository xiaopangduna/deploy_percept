#pragma once

#ifdef AWNN_FOUND

namespace deploy_percept
{
    namespace engine
    {

        /**
         * VIPLite 进程级 runtime（vip_init / vip_destroy）的 RAII 封装。
         *
         * 须在 AwnnEngine 之前构造、之后析构；同一进程内可多实例共享（引用计数）。
         */
        class VipLiteRuntime
        {
        public:
            VipLiteRuntime();
            ~VipLiteRuntime();

            VipLiteRuntime(const VipLiteRuntime &) = delete;
            VipLiteRuntime &operator=(const VipLiteRuntime &) = delete;
            VipLiteRuntime(VipLiteRuntime &&) = delete;
            VipLiteRuntime &operator=(VipLiteRuntime &&) = delete;

            bool ok() const { return ok_; }

        private:
            bool ok_{false};
            bool acquired_{false};
            static int ref_count_;
        };

    } // namespace engine
} // namespace deploy_percept

#endif
