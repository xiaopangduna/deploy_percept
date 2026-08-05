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
#ifndef __SGS_CMDLINE_H__
#define __SGS_CMDLINE_H__

#ifdef __cplusplus
extern "C" {
#endif

MI_S32 ST_Algo_ShowUsage(char *exeName, MI_FLOAT defaultThreshold);
MI_S32 ST_Algo_ParseCmdOptions(MI_S32 argc, char *argv[], std::string &inImages, char *modelPath, MI_FLOAT *threshold);

#ifdef __cplusplus
}
#endif

#endif