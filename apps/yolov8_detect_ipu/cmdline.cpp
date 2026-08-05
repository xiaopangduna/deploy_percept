/* Sgs trade secret */
/* Copyright (c) [2019~2025] Sgs Technology Ltd.
All rights reserved.

Unless otherwise stipulated in writing, any and all information contained
herein regardless in any format shall remain the sole proprietary of
Sgs and be kept in strict confidence
(Sgs Confidential Information) by the recipient.
Any unauthorized act including without limitation unauthorized disclosure,
copying, use, reproduction, sale, distribution, modification, disassembling,
reverse engineering and compiling of the contents of Sgs Confidential
Information is unlawful and strictly prohibited. Sgs hereby reserves the
rights to any and all damages, losses, costs and expenses resulting therefrom.
*/

#include "mi_ipu.h"
#include "mi_sys.h"

#include <getopt.h>
#include <cstdio>
#include <cstring>
#include <fstream>

#include "common.h"
#include "cmdline.h"

MI_S32 ST_Algo_ShowUsage(char *exeName, MI_FLOAT defaultThreshold)
{
    printf("usage: ./%s [-i/--images/--txts/--wavs] ([-h/--help] [-t/--threshold])\n", exeName);
    printf("-i, --images, --txts, --wavs    : required; path to input images (glob expression supported)\n");
    printf("-m, --model     : optional; path to model img, ignore while using builtin model\n");
    printf("-h, --help      : optional; show this help message and exit\n");
    printf("-t, --threshold : optional; set threshold(0.0~1.0), default=%f\n", defaultThreshold);
    return MI_SUCCESS;
}

MI_S32 ST_Algo_ParseCmdOptions(MI_S32 argc, char *argv[], std::string &inImages, char *modelPath, MI_FLOAT *threshold)
{

    struct option optionsCfg[] = {
        {"help", no_argument, NULL, 'h'},
        {"images/txts/wavs", required_argument, NULL, 'i'},
        {"threshold", optional_argument, NULL, 't'},
        {"model", optional_argument, NULL, 'm'},
    };

    MI_S32 s32OptIndex = 0;
    MI_S32 s32OptId = 0;

    MI_FLOAT defaultThreshold = *threshold;
    while (true)
    {
        s32OptId = getopt_long(argc, argv, "hi:t:m:", optionsCfg, &s32OptIndex);
        if (s32OptId == -1)
        {
            break;
        }

        switch (s32OptId)
        {
        case 'i':
            inImages = optarg;
            break;
        case 'm':
            if (strlen(optarg) >= MAX_PATH_STRLEN)
            {
                printf("%s\n", optarg);
                printf("model path is too long (max=%d\n", MAX_PATH_STRLEN);
                exit(__LINE__);
            }
            strcpy(modelPath, optarg);
            break;
        case 't':
            *threshold = atof(optarg);
            break;
        case 'h':
            ST_Algo_ShowUsage(argv[0], defaultThreshold);
            exit(0);
        default:
            ST_Algo_ShowUsage(argv[0], defaultThreshold);
            exit(1);
        }
    }

    if (inImages.size() != 0 && *threshold >= 0.0 && *threshold <= 1.0)
    {
        printf("inputs: %s\n", inImages.c_str());
        printf("model path: %s\n", modelPath);
        printf("threshold: %f\n", *threshold);
    }
    else
    {
        ST_Algo_ShowUsage(argv[0], defaultThreshold);
        exit(1);
    }

    return MI_SUCCESS;
}