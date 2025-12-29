#include "deploy_percept/engine/BaseEngine.hpp"
#include <cstdlib>
#include <cstdio>
#include <cstring>

namespace deploy_percept
{
    namespace engine
    {

        unsigned char *BaseEngine::load_data(FILE *fp, size_t ofst, size_t sz)
        {
            unsigned char *data;
            int ret;

            data = nullptr;

            if (nullptr == fp)
            {
                return nullptr;
            }

            ret = fseek(fp, ofst, SEEK_SET);
            if (ret != 0)
            {
                printf("blob seek failure.\n");
                return nullptr;
            }

            data = (unsigned char *)malloc(sz);
            if (data == nullptr)
            {
                printf("buffer malloc failure.\n");
                return nullptr;
            }
            ret = fread(data, 1, sz, fp);
            return data;
        }

        unsigned char *BaseEngine::load_model(const char *filename, int *model_size)
        {
            FILE *fp;
            unsigned char *data;

            fp = fopen(filename, "rb");
            if (nullptr == fp)
            {
                printf("Open file %s failed.\n", filename);
                return nullptr;
            }

            fseek(fp, 0, SEEK_END);
            int size = ftell(fp);
            if (size <= 0)
            {
                printf("Get file size failed.\n");
                fclose(fp);
                return nullptr;
            }

            rewind(fp); // 重置文件指针到开始位置
            data = load_data(fp, 0, size);

            fclose(fp);

            *model_size = size;
            return data;
        }

    } // namespace engine
} // namespace deploy_percept