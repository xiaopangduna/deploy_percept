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
#ifndef __SGS_COMMON_H__
#define __SGS_COMMON_H__

#ifdef __cplusplus
extern "C" {
#endif

#ifdef ALIGN_UP
#undef ALIGN_UP
#define ALIGN_UP(x, align) (((x) + ((align)-1)) & ~((align)-1))
#else
#define ALIGN_UP(x, align) (((x) + ((align)-1)) & ~((align)-1))
#endif
#ifndef ALIGN_DOWN
#define ALIGN_DOWN(val, alignment) (((val) / (alignment)) * (alignment))
#endif

#define ST_DEFAULT_SOC_ID 0
#define MAX_PATH_STRLEN 256
#define MAX_NUM_OUT_TENSOR 7
#define INNER_MOST_ALIGNMENT 1
#define DIVP_YUV_ALIGNMENT (16)
#define MAX_NUM_OUT_OBJS 255
#define MAX_MASK_PROTOS_CHANNEL 32
#define FRAME_LEN           (256)
#define MAX_VALUE_NUM       (128)
#define INPUT_LEGNTH        (114688)

typedef struct
{
    MI_U32 width;
    MI_U32 height;
    MI_IPU_ELEMENT_FORMAT format;
} ModelAttr_t;

typedef struct
{
    void *ptTensorData;
    MI_PHY phyTensorAddr;
    MI_U32 u32BufSize;
    MI_U64 u64Pts;
} AlgoInput_t;

typedef struct
{
    char riffHeader[4];         // Contains "RIFF"
    int wavSize;                // Size of the WAV file
    char waveHeader[4];         // Contains "WAVE"
    char fmtHeader[4];          // Contains "fmt " (with a space after fmt)
    int fmtChunkSize;          // Should be 16 for PCM format
    short audioFormat;          // Should be 1 for PCM format
    short numChannels;
    int sampleRate;
    int byteRate;               // Number of bytes per second. sample_rate * num_channels * Bytes Per Sample
    short sampleAlignment;      // num_channels * Bytes Per Sample
    short bitDepth;             // Number of bits per sample
    char dataHeader[4];         // Contains "data"
    int dataBytes;              // Number of bytes in data. Number of samples * num_channels * sample byte size
} WavHeader_t;

#ifdef __cplusplus
}
#endif

#endif