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

#pragma once
#ifndef __SGS_CLASSIFY_API_H__
#define __SGS_CLASSIFY_API_H__

#ifdef __cplusplus
extern "C" {
#endif
#include "common.h"

MI_S32 ST_Algo_InitBuffer(MI_U16 u16SocId, const ModelAttr_t *pModelAttr, AlgoInput_t *pAlgoInput);
MI_S32 ST_Algo_FillBuffer(std::string inputPath, ModelAttr_t *pModelAttr, const AlgoInput_t *pAlgoInput);
void ST_Algo_DumpBuffer(ModelAttr_t *modelAttr, const AlgoInput_t *algoInput, cv::Mat &dstMat);
void ST_Algo_WriteWav(std::string path, MI_FLOAT* samples, size_t size, MI_S32 sampleRate=8000);
void ST_Algo_GetFile(MI_S16 *ptr, const char *files);
bool ST_Algo_Create_Directory(const std::string& path) ;

#ifdef __cplusplus
}
#endif

#endif