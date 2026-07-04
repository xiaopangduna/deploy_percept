#pragma once

#include "deploy_percept/common/types.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace deploy_percept
{
    namespace engine
    {

        // ==================================================================
        // 推理引擎抽象基类
        //
        // 职责：定义 run 之后「如何取 raw 输出、如何释放」的统一协议。
        // 不约束 run() 签名（AWNN / RKNN 输入方式不同），也不持有具体 runtime 状态。
        //
        // 与 post_process 的分工：
        //   engine.getResult().tensor_views  →  模型 raw 输出（非 owning）
        //   processor.getResult()            →  业务结果（检测框、分类 id 等）
        // ==================================================================

        /**
         * 推理引擎基类。
         *
         * 子类在 run 成功后填充 getResult().tensor_views；调用方读完数据后须
         * releaseResult()，或由 EngineResultGuard 在析构时自动释放。
         */
        class BaseEngine
        {
        public:
            /**
             * 单次 run 的推理输出快照。
             * tensor_views 指向设备/NPU 映射内存，非 owning；有效至 releaseResult() 或下次 run()。
             */
            struct Result
            {
                std::vector<TensorView> tensor_views;
            };

            virtual ~BaseEngine() = default;

            /** 最近一次成功 run 的输出；release 后 tensor_views 为空 */
            virtual const Result &getResult() const = 0;

            /** 释放 mapped 输出并清空 getResult().tensor_views */
            virtual void releaseResult() = 0;

        protected:
            /** 读取二进制文件大小（供 RknnEngine 等从文件加载模型） */
            bool get_binary_file_size(const std::string &filepath, std::size_t &size);

            /** 将整个二进制文件读入 buffer（供 RknnEngine 等从文件加载模型） */
            bool load_binary_file_data(const std::string &filepath, std::vector<unsigned char> &data);
        };

    } // namespace engine
} // namespace deploy_percept
