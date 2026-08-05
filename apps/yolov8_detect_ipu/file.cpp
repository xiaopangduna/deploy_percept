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

#include <cstring>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <fcntl.h>

#include "file.h"
#include "common.h"

MI_S32 ST_Algo_OpenSourceFile(const char *pFileName, MI_S32 *pSrcFd)
{
    MI_S32 s32SrcFd = open(pFileName, O_RDONLY);
    if (s32SrcFd < 0)
    {
        return s32SrcFd;
    }
    *pSrcFd = s32SrcFd;

    return MI_SUCCESS;
}

MI_S32 ST_Algo_GetOneFrame(MI_S32 s32SrcFd, char *pData, MI_U32 u32YuvSize)
{
    off_t end = lseek(s32SrcFd, 0, SEEK_END);
    off_t start = lseek(s32SrcFd, 0, SEEK_SET);

    if (end - start >= u32YuvSize)
    {
        if (end - start > u32YuvSize)
        {
            printf("file content(%d) is more than buffer size(%d)\n", end - start, u32YuvSize);
        }

        MI_S32 s32ReadBytes = read(s32SrcFd, pData, u32YuvSize);

        return s32ReadBytes;
    }

    return -1;
}

MI_S32 ST_Algo_InitBuffer(MI_U16 u16SocId, const ModelAttr_t *pModelAttr, AlgoInput_t *pAlgoInput)
{

    memset(pAlgoInput, 0, sizeof(AlgoInput_t));

    MI_U32 u32InHeight = pModelAttr->height;
    MI_U32 u32InWidth = pModelAttr->width;
    MI_IPU_ELEMENT_FORMAT elementFormat = pModelAttr->format;
    MI_U32 u32BuffSize = 0;
    MI_U32 u32RowStride = 0;
    // printf("unknown element format %d\n", elementFormat);
    if (elementFormat == MI_IPU_ELEMENT_FORMAT::MI_IPU_FORMAT_U8 ||
        elementFormat == MI_IPU_ELEMENT_FORMAT::MI_IPU_FORMAT_INT8 ||
        elementFormat == MI_IPU_ELEMENT_FORMAT::MI_IPU_FORMAT_GRAY)
    {
        u32RowStride = u32InWidth;
        u32BuffSize = u32RowStride * u32InHeight;
    }
    else if (elementFormat == MI_IPU_ELEMENT_FORMAT::MI_IPU_FORMAT_NV12)
    {
        u32RowStride = ALIGN_UP(u32InWidth, DIVP_YUV_ALIGNMENT);
        u32BuffSize = (MI_U32)(u32RowStride * u32InHeight * 1.5);
    }
    else if (elementFormat == MI_IPU_ELEMENT_FORMAT::MI_IPU_FORMAT_INT16)
    {
        if(u32InWidth==0 && u32InHeight==0)
        {
            u32BuffSize=2;
        }
        else if(u32InWidth==0 && u32InHeight!=0)
        {
            u32BuffSize = u32InHeight * 2;
        }
        else
        {
            u32RowStride = u32InWidth * 2;
            u32BuffSize = u32RowStride * u32InHeight;
        }

    }
    else if (elementFormat == MI_IPU_ELEMENT_FORMAT::MI_IPU_FORMAT_INT32 ||
             elementFormat == MI_IPU_ELEMENT_FORMAT::MI_IPU_FORMAT_FP32 ||
             elementFormat == MI_IPU_ELEMENT_FORMAT::MI_IPU_FORMAT_ARGB8888 ||
             elementFormat == MI_IPU_ELEMENT_FORMAT::MI_IPU_FORMAT_ABGR8888)
    {
        if(u32InWidth==0 && u32InHeight==0)
        {
            u32BuffSize=4;
        }
        else if(u32InWidth==0 && u32InHeight!=0)
        {
            u32BuffSize = u32InHeight * 4;
        }
        else
        {
            u32RowStride = u32InWidth * 4;
            u32BuffSize = u32RowStride * u32InHeight;
        }
    }
    else
    {
        // printf("unknown element format %d\n", elementFormat);
        // exit(__LINE__);
        u32BuffSize = u32InHeight * 2;
    }

    MI_PHY  phyTensorAddr = 0;
    void * ptTensorData = NULL;

    if (pAlgoInput->u32BufSize != 0)
    {
        printf("pAlgoInput should refer to an uninitialized buffer !\n");
    }

    MI_S32 ret = MI_SYS_MMA_Alloc(u16SocId, NULL, u32BuffSize, &phyTensorAddr);
    if (ret != MI_SUCCESS)
    {
        printf("alloc src buff failed\n");
        exit(__LINE__);
    }

    ret = MI_SYS_Mmap(phyTensorAddr, u32BuffSize, &ptTensorData, TRUE);
    if (ret != MI_SUCCESS)
    {
        printf("mmap src buff failed\n");
        exit(__LINE__);
    }

    memset(ptTensorData, 0, u32BuffSize);

    pAlgoInput->u32BufSize = u32BuffSize;
    pAlgoInput->phyTensorAddr = phyTensorAddr;
    pAlgoInput->ptTensorData = ptTensorData;

    return MI_SUCCESS;
}

void ST_Algo_DumpBuffer(ModelAttr_t *inputAttr, const AlgoInput_t *algoInput, cv::Mat &dstMat)
{
    MI_U8 *pVirsrc = (MI_U8 *)algoInput->ptTensorData;
    MI_U32 inWidth = inputAttr->width;
    MI_U32 inHeight = inputAttr->height;
    MI_IPU_ELEMENT_FORMAT inFormat = inputAttr->format;

    if ((MI_U32) dstMat.rows == inHeight && (MI_U32) dstMat.cols == inWidth && dstMat.channels() == 3)
    {
        // do nothing
    }
    else
    {
        if (dstMat.empty() == false)
        {
            dstMat.release();
        }

        dstMat.create(inHeight, inWidth, CV_8UC3);
    }

    if (inFormat == MI_IPU_ELEMENT_FORMAT::MI_IPU_FORMAT_ARGB8888)
    {
        cv::Mat bgra(inHeight, inWidth, CV_8UC4);
        memcpy(bgra.data, (unsigned char *)pVirsrc, inHeight * inWidth * 4 * sizeof(unsigned char));
        cv::cvtColor(bgra, dstMat, cv::COLOR_BGRA2BGR);
    }
    else if (inFormat == MI_IPU_ELEMENT_FORMAT::MI_IPU_FORMAT_ABGR8888)
    {
        cv::Mat rgba(inHeight, inWidth, CV_8UC4);
        memcpy(rgba.data, (unsigned char *)pVirsrc, inHeight * inWidth * 4 * sizeof(unsigned char));
        cv::cvtColor(rgba, dstMat, cv::COLOR_RGBA2BGR);
    }
    else if (inFormat == MI_IPU_ELEMENT_FORMAT::MI_IPU_FORMAT_NV12)
    {
        cv::Mat yuv(inHeight * 3 / 2, inWidth, CV_8UC1);
        memcpy(yuv.data, (unsigned char *)pVirsrc, inHeight * 3 / 2 * inWidth * sizeof(unsigned char));
        cv::cvtColor(yuv, dstMat, cv::COLOR_YUV2BGR_NV12);
    }
    else
    {
        std::cout << "not supported element format " << inFormat
                  << " for dump " << std::endl;
        exit(__LINE__);
    }
}

MI_BOOL ST_Algo_EndsWith(std::string const &fullStr, std::string const &ending)
{
    if (fullStr.length() >= ending.length())
    {
        return (0 == fullStr.compare(fullStr.length() - ending.length(), ending.length(), ending));
    }
    else
    {
        return false;
    }
}

MI_S32 ST_Algo_YV12toNV12(const cv::Mat &input, cv::Mat &output)
{
    MI_U32 u32Width = input.cols;
    MI_U32 u32Height = input.rows * 2 / 3;
    MI_U32 u32Stride = (MI_U32)input.step[0]; // Rows bytes stride - in most cases equal to width

    input.copyTo(output);

    // Y Channel
    //  YYYYYYYYYYYYYYYY
    //  YYYYYYYYYYYYYYYY
    //  YYYYYYYYYYYYYYYY
    //  YYYYYYYYYYYYYYYY
    //  YYYYYYYYYYYYYYYY
    //  YYYYYYYYYYYYYYYY

    // V Input channel
    //  VVVVVVVV
    //  VVVVVVVV
    //  VVVVVVVV
    cv::Mat inV = cv::Mat(cv::Size(u32Width / 2, u32Height / 2), CV_8UC1, (char *)input.data + u32Stride * u32Height, u32Stride / 2); // Input V color channel (in YV12 V is above U).

    // U Input channel
    //  UUUUUUUU
    //  UUUUUUUU
    //  UUUUUUUU
    cv::Mat inU = cv::Mat(cv::Size(u32Width / 2, u32Height / 2), CV_8UC1, (char *)input.data + u32Stride * u32Height + (u32Stride / 2) * (u32Height / 2), u32Stride / 2); // Input V color channel (in YV12 U is below V).

    for (MI_U32 u32Row = 0; u32Row < u32Height / 2; u32Row++)
    {
        for (MI_U32 u32Col = 0; u32Col < u32Width / 2; u32Col++)
        {
            output.at<uchar>(u32Height + u32Row, 2 * u32Col) = inU.at<uchar>(u32Row, u32Col);
            output.at<uchar>(u32Height + u32Row, 2 * u32Col + 1) = inV.at<uchar>(u32Row, u32Col);
        }
    }

    return MI_SUCCESS;
}

MI_S32 ST_Algo_FillBuffer(std::string inputPath, ModelAttr_t *pModelAttr, const AlgoInput_t *pAlgoInput)
{
    char *pVirsrc = (char *)pAlgoInput->ptTensorData;
    MI_U32 u32InWidth = pModelAttr->width;
    MI_U32 u32InHeight = pModelAttr->height;
    MI_IPU_ELEMENT_FORMAT inFormat = pModelAttr->format;
    printf("fillbuffer processing...\n");

    if (ST_Algo_EndsWith(inputPath, std::string(".jpg")) || ST_Algo_EndsWith(inputPath, std::string(".png")) ||
        ST_Algo_EndsWith(inputPath, std::string(".bmp")))
    {
        cv::Mat rawBgr = cv::imread(inputPath);

        cv::Mat resizedBgr;

        if (u32InWidth != rawBgr.cols || u32InHeight != rawBgr.rows)
        {
            cv::resize(rawBgr, resizedBgr, cv::Size(u32InWidth, u32InHeight), 0, 0, cv::INTER_LINEAR);
            printf("net input width: %d, net input height: %d\n", u32InWidth, u32InHeight);
        }
        else
        {
            resizedBgr = rawBgr;
        }

        if (inFormat == MI_IPU_ELEMENT_FORMAT::MI_IPU_FORMAT_ARGB8888)
        {
            cv::Mat resizedBgra;
            cv::cvtColor(resizedBgr, resizedBgra, cv::COLOR_BGR2BGRA);

            MI_U32 u32RowStride = ALIGN_UP(u32InWidth * 4, DIVP_YUV_ALIGNMENT);
            MI_U32 u32BuffSize = u32InHeight * u32RowStride;
            for (MI_U32 u32Row = 0; u32Row < u32InHeight; u32Row++)
            {
                memcpy(pVirsrc, resizedBgra.ptr<uchar>(u32Row), u32InWidth * 4);
                pVirsrc += u32RowStride;
            }
        }
        else if (inFormat == MI_IPU_ELEMENT_FORMAT::MI_IPU_FORMAT_ABGR8888)
        {
            cv::Mat resizedRgba;
            cv::cvtColor(resizedBgr, resizedRgba, cv::COLOR_BGR2RGBA);

            MI_U32 u32RowStride = ALIGN_UP(u32InWidth * 4, DIVP_YUV_ALIGNMENT);
            MI_U32 u32BuffSize = u32InHeight * u32RowStride;
            for (MI_U32 u32Row = 0; u32Row < u32InHeight; u32Row++)
            {
                memcpy(pVirsrc, resizedRgba.ptr<uchar>(u32Row), u32InWidth * 4);
                pVirsrc += u32RowStride;
            }
        }
        else if (inFormat == MI_IPU_ELEMENT_FORMAT::MI_IPU_FORMAT_NV12)
        {

            MI_U32 u32AlignHeight = ALIGN_DOWN(resizedBgr.rows, 2);
            MI_U32 u32AlignWidth = ALIGN_DOWN(resizedBgr.cols, 2);
            MI_U32 u32NumCells = u32AlignHeight * u32AlignWidth * 3 / 2;
            cv::Mat alignBgr = resizedBgr(cv::Range(0, u32AlignHeight), cv::Range(0, u32AlignWidth)).clone();
            cv::Mat alignYv12;
            cv::cvtColor(alignBgr, alignYv12, cv::COLOR_BGR2YUV_YV12);
            cv::Mat alignYuv = alignYv12.clone();
            ST_Algo_YV12toNV12(alignYv12, alignYuv);

            MI_U8 *pSrc = alignYuv.data;
            MI_U32 u32RowStride = ALIGN_UP(u32InWidth, DIVP_YUV_ALIGNMENT);
            for (MI_U32 i = 0; i < alignYuv.rows; i++)
            {
                memcpy(pVirsrc, pSrc, u32InWidth);
                pVirsrc += u32RowStride;
                pSrc += u32InWidth;
            }
        }
        else if (inFormat == MI_IPU_ELEMENT_FORMAT::MI_IPU_FORMAT_FP32)
        {
            cv::Mat resizedGray;
            cv::cvtColor(resizedBgr, resizedGray, cv::COLOR_BGR2GRAY);

            MI_U32 u32RowStride = ALIGN_UP(u32InWidth * 4, DIVP_YUV_ALIGNMENT);
            MI_U32 u32BuffSize = u32InHeight * u32RowStride;
            for (MI_U32 u32Row = 0; u32Row < u32InHeight; u32Row++)
            {
                memcpy(pVirsrc, resizedGray.ptr<uchar>(u32Row), u32InWidth * 4);
                pVirsrc += u32RowStride;
            }
        }
        else
        {
            printf("not supported element format %d for input%s\n", inFormat, inputPath);
            exit(__LINE__);
        }
    }
    else if (ST_Algo_EndsWith(inputPath, std::string(".argb")) || ST_Algo_EndsWith(inputPath, std::string(".bgra")) ||
             (ST_Algo_EndsWith(inputPath, std::string(".bin")) && (inFormat == MI_IPU_ELEMENT_FORMAT::MI_IPU_FORMAT_ARGB8888)))
    {
        if (inFormat == MI_IPU_ELEMENT_FORMAT::MI_IPU_FORMAT_ARGB8888)
        {
            MI_S32 s32SrcFd;
            MI_S32 s32Ret = ST_Algo_OpenSourceFile(inputPath.c_str(), &s32SrcFd);
            if (s32Ret != MI_SUCCESS)
            {
                printf("open file[%s] failed\n", inputPath.c_str());
                exit(__LINE__);
            }

            MI_U32 u32RowStride = ALIGN_UP(u32InWidth * 4, DIVP_YUV_ALIGNMENT);
            MI_U32 u32BuffSize = u32InHeight * u32RowStride;

            MI_U32 u32NumWriteBytes = ST_Algo_GetOneFrame(s32SrcFd, (char *)pVirsrc, u32BuffSize);
            if (u32NumWriteBytes == u32BuffSize)
            {
                close(s32SrcFd);
                MI_SYS_FlushInvCache(pVirsrc, u32BuffSize);
            }
            else
            {
                printf("file2buffer failed, expected %d bytes, %d bytes written\n", u32BuffSize, u32NumWriteBytes);
                exit(__LINE__);
            }
        }
        else
        {
            printf("not supported element format %d for input%s\n", inFormat, inputPath.c_str());
            exit(__LINE__);
        }
    }
    else if (ST_Algo_EndsWith(inputPath, std::string(".yuv")) ||
             (ST_Algo_EndsWith(inputPath, std::string(".bin")) && (inFormat == MI_IPU_ELEMENT_FORMAT::MI_IPU_FORMAT_NV12)))
    {
        if (inFormat == MI_IPU_ELEMENT_FORMAT::MI_IPU_FORMAT_NV12)
        {
            MI_S32 s32SrcFd;
            MI_S32 s32Ret = ST_Algo_OpenSourceFile(inputPath.c_str(), &s32SrcFd);
            if (s32Ret < 0)
            {
                printf("open file[%s] fail\n", inputPath.c_str());
                exit(__LINE__);
            }

            MI_U32 u32RowStride = ALIGN_UP(u32InWidth, DIVP_YUV_ALIGNMENT);
            MI_U32 u32BuffSize = u32InHeight * u32RowStride * 3 / 2;
            // MI_S16 u32BuffSize = u32InHeight * u32InWidth * 4;

            MI_U32 u32NumWriteBytes = ST_Algo_GetOneFrame(s32SrcFd, (char *)pVirsrc, u32BuffSize);
            if (u32NumWriteBytes == u32BuffSize)
            {
                close(s32SrcFd);
                MI_SYS_FlushInvCache(pVirsrc, u32BuffSize);
            }
            else
            {
                printf("file2buffer failed, expected %d bytes, %d bytes written\n", u32BuffSize, u32NumWriteBytes);
                exit(__LINE__);
            }
        }
        else
        {
            printf("not supported element format %d for input%s\n", inFormat, inputPath.c_str());
            exit(__LINE__);
        }
    }
    return MI_SUCCESS;
}

void ST_Algo_WriteWav(std::string path, MI_FLOAT* samples, size_t size, MI_S32 sampleRate)
{
    WavHeader_t wavHeader;
    // MI_S32 sampleRate = 8000;
    MI_S32 numChannels = 1;
    MI_S32 bitDepth = 16;

    std::vector<short> pcmSamples(size);
    for (size_t i = 0; i < size; ++i) {
        pcmSamples[i] = static_cast<short>(std::max(-1.0f, std::min(1.0f, samples[i])) * 32767);
    }

    // Open file
    std::ofstream file(path, std::ios::binary);

    // Write the WAV header
    memcpy(wavHeader.riffHeader, "RIFF", 4);
    memcpy(wavHeader.waveHeader, "WAVE", 4);
    memcpy(wavHeader.fmtHeader, "fmt ", 4);
    wavHeader.fmtChunkSize = 16;
    wavHeader.audioFormat = 1;
    wavHeader.numChannels = numChannels;
    wavHeader.sampleRate = sampleRate;
    wavHeader.byteRate = sampleRate * numChannels * (bitDepth / 8);
    wavHeader.sampleAlignment = numChannels * (bitDepth / 8);
    wavHeader.bitDepth = bitDepth;
    memcpy(wavHeader.dataHeader, "data", 4);
    wavHeader.dataBytes = pcmSamples.size() * (bitDepth / 8);
    wavHeader.wavSize = 4 + (8 + wavHeader.fmtChunkSize) + (8 + wavHeader.dataBytes);

    // Write header to file
    file.write(reinterpret_cast<const char*>(&wavHeader), sizeof(WavHeader_t));

    // Write audio samples
    file.write(reinterpret_cast<const char*>(pcmSamples.data()), wavHeader.dataBytes);

    // Close file
    file.close();

    std::cout << "WAV file '" << path << "' has been written" << std::endl;
}

void ST_Algo_GetFile(MI_S16 *ptr, const char *files)
{
    std::ifstream infile;
    infile.open(files);

    while(!infile.eof())
    {
        infile>>*ptr;
        ptr++;
    }
    infile.close();
}

bool ST_Algo_Create_Directory(const std::string& path) 
{
    std::string temp_path;
    size_t pos = 0;

    while ((pos = path.find('/', pos)) != std::string::npos) {
        temp_path = path.substr(0, pos++);
        if (temp_path.empty()) continue;

        if (mkdir(temp_path.c_str(), 0755) && errno != EEXIST) {
            std::cerr << "Error creating directory: " << strerror(errno) << std::endl;
            return false;
        }
    }

    if (mkdir(path.c_str(), 0755) && errno != EEXIST) {
        std::cerr << "Error creating directory: " << strerror(errno) << std::endl;
        return false;
    }

    return true;
}