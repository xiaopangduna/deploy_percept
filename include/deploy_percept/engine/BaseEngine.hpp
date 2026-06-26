#pragma once

#include "deploy_percept/post_process/types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace deploy_percept
{
    namespace engine
    {

        /**
         * 推理引擎公共基类。
         *
         * 统一约束输出借还（borrow / release），供 OutputAccess 等多态使用。
         * run() 由各子类自行定义（参数、数量不强制一致）。
         *
         * 典型流程：engine.run(...); → OutputAccess out(engine); → post(out.views());
         */
        class BaseEngine
        {
        public:
            virtual ~BaseEngine() = default;

            /** run 成功后借出输出 TensorView；有效至 release_output_views() 或下次 run() */
            virtual std::vector<post_process::TensorView> borrow_output_views() const = 0;

            /** 归还借出的 views（Mapped / RKNN 等有实际释放；HostCopy 可为 no-op） */
            virtual void release_output_views() = 0;

        protected:
            bool get_binary_file_size(const std::string &filepath, std::size_t &size);

            bool load_binary_file_data(const std::string &filepath, std::vector<unsigned char> &data);
        };

    } // namespace engine
} // namespace deploy_percept
