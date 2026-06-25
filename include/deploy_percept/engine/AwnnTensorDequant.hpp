#pragma once

#ifdef AWNN_FOUND

#include "deploy_percept/engine/TensorDesc.hpp"

#include <cstddef>
#include <vector>

namespace deploy_percept
{
    namespace engine
    {

        class AwnnTensorDequant
        {
        public:
            static void buildQuantLut(TensorDesc &desc);

            static bool toFloat(
                const TensorDesc &desc,
                const void *raw_data,
                std::size_t raw_byte_size,
                std::vector<float> &out_float);
        };

    } // namespace engine
} // namespace deploy_percept

#endif
