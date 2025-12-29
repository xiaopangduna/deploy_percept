#pragma once

#include <cstdio>
#include <cstddef>
#include <cstdint>

namespace deploy_percept
{
    namespace engine
    {

        class BaseEngine
        {
        public:
            static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
            static unsigned char *load_model(const char *filename, int *model_size);
        };

    } // namespace engine
} // namespace deploy_percept