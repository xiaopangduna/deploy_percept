/**
 * Sigmastar IPU YOLOv8 检测 demo（复现 OpenDLA source/detection/yolov8）
 *
 * 用法：
 *   yolov8_detect_ipu -i "./images/*.jpg" -m yolov8n_640x640.img [-t 0.5]
 *
 * 原始来源：
 *   tmp/.../sdk/verify/opendla/source/detection/yolov8/yolov8.cpp
 */
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
#include <unistd.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "mi_common_datatype.h"
#include "mi_sys_datatype.h"
#include "mi_ipu.h"
#include "mi_sys.h"

#include "file.h"
#include "cmdline.h"
#include "common.h"

typedef struct
{
    void *data[MAX_NUM_OUT_TENSOR] = {0};
    MI_FLOAT scalar[MAX_NUM_OUT_TENSOR] = {0};
    MI_IPU_ELEMENT_FORMAT outFormat;
    MI_U32 netInW;
    MI_U32 netInH;
    MI_U32 outShapes[MAX_NUM_OUT_TENSOR][MI_IPU_MAX_TENSOR_DIM] = {0};
    MI_U32 numOutLayers;
    MI_S32 numClasses;
    MI_FLOAT confThreshold;
    MI_FLOAT nmsThreshold;
    std::vector<MI_S32> strides;
    MI_S32 regMax=16;
} DetParams_t;

typedef struct BoxInfo_t
{
    MI_FLOAT x1;
    MI_FLOAT y1;
    MI_FLOAT x2;
    MI_FLOAT y2;
    MI_FLOAT score;
    MI_S16 label;
    MI_U64 pts;
} BoxInfo_t;

std::vector<std::string> cocoClasses =
    {
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush"};


void ST_ALGO_CheckBoxes(std::vector<BoxInfo_t> bboxes, MI_S32 width, MI_S32 height)
{
    for (size_t i = 0; i < bboxes.size(); i++)
    {
        BoxInfo_t *rBox = &(bboxes[i]);

        if (rBox->x1 < 0)
        {
            rBox->x1 = 0;
        }
        else if (rBox->x1 > width - 1)
        {
            rBox->x1 = width - 1;
        }
        if (rBox->x2 < 0)
        {
            rBox->x2 = 0;
        }
        else if (rBox->x2 > width - 1)
        {
            rBox->x2 = width - 1;
        }
        if (rBox->x1 > rBox->x2)
        {
            rBox->x1 = rBox->x2;
        }

        if (rBox->y1 < 0)
        {
            rBox->y1 = 0;
        }
        else if (rBox->y1 > height - 1)
        {
            rBox->y1 = height - 1;
        }
        if (rBox->y2 < 0)
        {
            rBox->y2 = 0;
        }
        else if (rBox->y2 > height - 1)
        {
            rBox->y2 = height - 1;
        }
        if (rBox->y1 > rBox->y2)
        {
            rBox->y1 = rBox->y2;
        }

        if (rBox->score < 0.0)
        {
            rBox->score = 0.0;
        }
        else if (rBox->score > 1.0)
        {
            rBox->score = 1.0;
        }
    }
}


void ST_ALGO_Nms(std::vector<BoxInfo_t> &inputBoxes, MI_FLOAT NMS_THRESH)
{
    std::sort(inputBoxes.begin(), inputBoxes.end(), [](BoxInfo_t a, BoxInfo_t b)
              { return a.score > b.score; });
    std::vector<MI_FLOAT> vArea(inputBoxes.size());
    for (int i = 0; i < int(inputBoxes.size()); ++i)
    {
        vArea[i] = (inputBoxes[i].x2 - inputBoxes[i].x1 + 1) * (inputBoxes[i].y2 - inputBoxes[i].y1 + 1);
    }
    for (int i = 0; i < int(inputBoxes.size()); ++i)
    {
        for (int j = i + 1; j < int(inputBoxes.size());)
        {
            MI_FLOAT xx1 = std::max(inputBoxes[i].x1, inputBoxes[j].x1);
            MI_FLOAT yy1 = std::max(inputBoxes[i].y1, inputBoxes[j].y1);
            MI_FLOAT xx2 = std::min(inputBoxes[i].x2, inputBoxes[j].x2);
            MI_FLOAT yy2 = std::min(inputBoxes[i].y2, inputBoxes[j].y2);
            MI_FLOAT w = std::max(0.0f, xx2 - xx1 + 1);
            MI_FLOAT h = std::max(0.0f, yy2 - yy1 + 1);
            MI_FLOAT inter = w * h;
            MI_FLOAT ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH)
            {
                inputBoxes.erase(inputBoxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}


MI_S32 ST_ALGO_DetIpuPostProcessOneLayer(DetParams_t &params, int iLayer, std::vector<BoxInfo_t> &outBoxes)
{
    if ((MI_U32)iLayer >= params.numOutLayers)
    {
        printf("illegal access to out layer%d \n", iLayer);
        exit(__LINE__);
    }

    params.numClasses =  params.outShapes[iLayer][1] - 4; // auto fill numClasses

    MI_FLOAT cx, cy, w, h;

    std::vector<MI_S32> fg_box_indexes;

    if (params.outFormat == MI_IPU_ELEMENT_FORMAT::MI_IPU_FORMAT_FP32)
    {
        MI_FLOAT *cxHead = (MI_FLOAT *)params.data[iLayer];
        MI_FLOAT *cyHead = (MI_FLOAT *)params.data[iLayer] + 1 * params.outShapes[iLayer][2];
        MI_FLOAT *wHead = (MI_FLOAT *)params.data[iLayer] + 2 * params.outShapes[iLayer][2];
        MI_FLOAT *hHead = (MI_FLOAT *)params.data[iLayer] + 3 * params.outShapes[iLayer][2];
        MI_FLOAT *scoreHead;

        for (MI_U32 iBox = 0; iBox < params.outShapes[iLayer][2]; iBox++)
        {
            scoreHead = (MI_FLOAT *)params.data[iLayer] + 4 * params.outShapes[iLayer][2] + iBox;

            MI_FLOAT maxScore = -INFINITY;
            MI_S32 maxInd = -1;

            for (MI_S32 iClass = 0; iClass < params.numClasses; iClass++)
            {
                if (*scoreHead >= params.confThreshold && *scoreHead > maxScore) {
                    maxScore = *scoreHead;
                    maxInd = iClass;
                }
                scoreHead += params.outShapes[iLayer][2];
            }

            if (maxInd >= 0)
            {
                cx = cxHead[iBox];
                cy = cyHead[iBox];
                w = wHead[iBox];
                h = hHead[iBox];
                BoxInfo_t rBoxInfo;
                rBoxInfo.x1 = cx - w / 2.0;
                rBoxInfo.y1 = cy - h / 2.0;
                rBoxInfo.x2 = cx + w / 2.0;
                rBoxInfo.y2 = cy + h / 2.0;
                rBoxInfo.score = maxScore;
                rBoxInfo.label = maxInd;
                outBoxes.push_back(rBoxInfo);
            }
        }
    }
    else if (params.outFormat == MI_IPU_ELEMENT_FORMAT::MI_IPU_FORMAT_INT16)
    {
        MI_S16 *cxHead =  (MI_S16 *)params.data[iLayer];
        MI_S16 *cyHead =  (MI_S16 *)params.data[iLayer] + 1 * params.outShapes[iLayer][2];
        MI_S16 *wHead =   (MI_S16 *)params.data[iLayer] + 2 * params.outShapes[iLayer][2];
        MI_S16 *hHead =   (MI_S16 *)params.data[iLayer] + 3 * params.outShapes[iLayer][2];
        MI_S16 *scoreHead;

        MI_FLOAT threshold_before_scale = params.confThreshold / params.scalar[iLayer];

        if (params.scalar[iLayer] > 0)
        {
            for (MI_U32 iBox = 0; iBox < params.outShapes[iLayer][2]; iBox++)
            {
                scoreHead = (MI_S16 *)params.data[iLayer] + 4 * params.outShapes[iLayer][2] + iBox;

                MI_FLOAT maxScore = -INFINITY;
                MI_S32 maxInd = -1;

                for (MI_S32 iClass = 0; iClass < params.numClasses; iClass++)
                {
                    if (*scoreHead >= threshold_before_scale && *scoreHead > maxScore)
                    {
                        maxScore = *scoreHead;
                        maxInd = iClass;
                    }
                    scoreHead += params.outShapes[iLayer][2];
                }

                if (maxInd >= 0)
                {
                    cx = cxHead[iBox] * params.scalar[iLayer];
                    cy = cyHead[iBox] * params.scalar[iLayer];
                    w = wHead[iBox] * params.scalar[iLayer];
                    h = hHead[iBox] * params.scalar[iLayer];

                    BoxInfo_t rBoxInfo;
                    rBoxInfo.x1 = cx - w / 2.0;
                    rBoxInfo.y1 = cy - h / 2.0;
                    rBoxInfo.x2 = cx + w / 2.0;
                    rBoxInfo.y2 = cy + h / 2.0;
                    rBoxInfo.score = maxScore * params.scalar[iLayer];
                    rBoxInfo.label = maxInd;
                    outBoxes.push_back(rBoxInfo);
                }
            }
        }
        else
        {
            for (MI_U32 iBox = 0; iBox < params.outShapes[iLayer][2]; iBox++)
            {
                scoreHead = (MI_S16 *)params.data[iLayer] + 4 * params.outShapes[iLayer][2] + iBox;

                MI_FLOAT minScore = INFINITY;
                MI_S32 minInd = -1;

                for (MI_S32 iClass = 0; iClass < params.numClasses; iClass++)
                {
                    if (*scoreHead <= threshold_before_scale && *scoreHead < minScore)
                    {
                        minScore = *scoreHead;
                        minInd = iClass;
                    }
                    scoreHead += params.outShapes[iLayer][2];
                }

                if (minInd >= 0)
                {
                    cx = cxHead[iBox] * params.scalar[iLayer];
                    cy = cyHead[iBox] * params.scalar[iLayer];
                    w = wHead[iBox] * params.scalar[iLayer];
                    h = hHead[iBox] * params.scalar[iLayer];

                    BoxInfo_t rBoxInfo;
                    rBoxInfo.x1 = cx - w / 2.0;
                    rBoxInfo.y1 = cy - h / 2.0;
                    rBoxInfo.x2 = cx + w / 2.0;
                    rBoxInfo.y2 = cy + h / 2.0;
                    rBoxInfo.score = minScore * params.scalar[iLayer];
                    rBoxInfo.label = minInd;
                    outBoxes.push_back(rBoxInfo);
                }
            }
        }
    }
    else
    {
        printf("Not implemented out format: %d !\n", params.outFormat);
        exit(__LINE__);
    }


    return MI_SUCCESS;
}


static MI_FLOAT sigmoid(MI_FLOAT x)
{
    // return 1.0f / (1.0f + fast_exp(-x));
    return (1 / (1 + exp(-x)));
}


static void softmax_(const MI_FLOAT *x, MI_FLOAT *y, int length)
{
    MI_FLOAT sum = 0;
    int i = 0;
    for (i = 0; i < length; i++)
    {
        y[i] = exp(x[i]);
        sum += y[i];
    }
    for (i = 0; i < length; i++)
    {
        y[i] /= sum;
    }
}


MI_S32 ST_ALGO_DetCpuPostProcessOneLayer(DetParams_t &params, int iLayer, std::vector<BoxInfo_t> &outBoxes)
{
    if ((MI_U32)iLayer >= params.numOutLayers)
    {
        printf("illegal access to out layer%d \n", iLayer);
        exit(__LINE__);
    }



    params.regMax = 16;
    MI_U32 numChannels = params.outShapes[iLayer][3];


    params.numClasses = numChannels - params.regMax * 4;

    if (params.numClasses <= 0)
    {
        printf("numClasses should not be %d ! make sure regMax(%d) is set correctly\n", params.numClasses, params.regMax);
        exit(__LINE__);
    }


    MI_U32 gridW = params.outShapes[iLayer][2];
    MI_U32 gridH = params.outShapes[iLayer][1];

    MI_U32 strideW = params.netInW / gridW;
    MI_U32 strideH = params.netInH / gridH;

    if (strideW != strideH)
    {
        printf("strideW(%u) should equals to strideH(%u), make sure out tensor shape is correct\n", strideW, strideH);
        exit(__LINE__);
    }




    MI_FLOAT *scores = new MI_FLOAT[params.numClasses];
    MI_FLOAT *distribution = new MI_FLOAT[params.regMax];

    MI_FLOAT ltrb[4];

    const MI_FLOAT *pScore;

    MI_FLOAT thresholdBeforeSigmoid = log(params.confThreshold / (1.0 - params.confThreshold)); // y = sigmoid(x) = 1 / (1 + e^-x) ==> x = ln(y/(1-y))


    if (params.outFormat == MI_IPU_ELEMENT_FORMAT::MI_IPU_FORMAT_FP32)
    {
        printf("executing FP32 decode...\n");
        const MI_FLOAT *predsFloat = (MI_FLOAT *)params.data[iLayer];
        for (MI_U32 i = 0; i < gridH; i++)
        {
            for (MI_U32 j = 0; j < gridW; j++)
            {
                int maxInd = 0;
                MI_FLOAT maxScore = -INFINITY; // -inf

                pScore = &(predsFloat[4 * params.regMax]);


                for (int k = 0; k < params.numClasses; k++)
                {
                    if (pScore[k]> maxScore)
                    {
                        maxScore = pScore[k];
                        maxInd = k;
                    }
                }
                if (maxScore >= thresholdBeforeSigmoid)
                {
                    maxScore = sigmoid(maxScore);

                    const MI_FLOAT *pBox = predsFloat;

                    for (int k = 0; k < 4; k++)
                    {
                        softmax_(pBox + k * params.regMax, distribution, params.regMax);

                        ltrb[k] = 0.0;
                        for (int l = 0; l < params.regMax; l++)
                        {
                            ltrb[k] += l * distribution[l];
                        }
                        ltrb[k] = ltrb[k] * strideW;
                    }

                    MI_FLOAT pbCx = (j + 0.5) * strideW;
                    MI_FLOAT pbCy = (i + 0.5) * strideH;

                    BoxInfo_t rBoxInfo;
                    rBoxInfo.x1 = pbCx - ltrb[0];
                    rBoxInfo.x2 = pbCx + ltrb[2];
                    rBoxInfo.y1 = pbCy - ltrb[1];
                    rBoxInfo.y2 = pbCy + ltrb[3];
                    rBoxInfo.score = maxScore;
                    rBoxInfo.label = maxInd;
                    outBoxes.push_back(rBoxInfo);
                }
                predsFloat += ALIGN_UP(numChannels, INNER_MOST_ALIGNMENT);
            }
        }
    }
    else if (params.outFormat == MI_IPU_ELEMENT_FORMAT::MI_IPU_FORMAT_INT16)
    {
        printf("executing INT16 decode...\n");
        const MI_S16 *predsInt16 = (MI_S16 *)params.data[iLayer];
        MI_FLOAT scalar = params.scalar[iLayer];
        MI_FLOAT rScore;
        MI_FLOAT *dist_before_softmax = new MI_FLOAT[params.regMax];

        if (scalar >= 0)
        {
            for (MI_U32 i = 0; i < gridH; i++)
            {
                for (MI_U32 j = 0; j < gridW; j++)
                {

                    int maxInd = 0;
                    MI_FLOAT maxScore = -INFINITY; // -inf

                    for (int k = 0; k < params.numClasses; k++)
                    {
                        rScore = predsInt16[4 * params.regMax + k];
                        if (rScore > maxScore)
                        {
                            maxScore = rScore;
                            maxInd = k;
                        }
                    }

                    maxScore *= scalar;

                    if (maxScore >= thresholdBeforeSigmoid)
                    {
                        maxScore = sigmoid(maxScore );

                        const MI_S16 *pBox = predsInt16;

                        for (int k = 0; k < 4; k++)
                        {
                            for (int i_reg = 0; i_reg < params.regMax; i_reg++)
                            {
                                dist_before_softmax[i_reg] = *(pBox + k * params.regMax + i_reg) * params.scalar[iLayer];
                            }

                            softmax_(dist_before_softmax, distribution, params.regMax);

                            ltrb[k] = 0.0;
                            for (int l = 0; l < params.regMax; l++)
                            {
                                ltrb[k] += l * distribution[l];
                            }
                            ltrb[k] = ltrb[k] * strideW;
                        }

                        MI_FLOAT pbCx = (j + 0.5) * strideW;
                        MI_FLOAT pbCy = (i + 0.5) * strideH;

                        BoxInfo_t rBoxInfo;
                        rBoxInfo.x1 = pbCx - ltrb[0];
                        rBoxInfo.x2 = pbCx + ltrb[2];
                        rBoxInfo.y1 = pbCy - ltrb[1];
                        rBoxInfo.y2 = pbCy + ltrb[3];
                        rBoxInfo.score = maxScore;
                        rBoxInfo.label = maxInd;
                        outBoxes.push_back(rBoxInfo);
                    }
                    predsInt16 += ALIGN_UP(numChannels, INNER_MOST_ALIGNMENT);
                }
            }
        }
        else
        {
            for (MI_U32 i = 0; i < gridH; i++)
            {
                for (MI_U32 j = 0; j < gridW; j++)
                {

                    int maxInd = 0;
                    MI_FLOAT maxScore = -INFINITY; // -inf

                    for (int k = 0; k < params.numClasses; k++)
                    {
                        rScore = predsInt16[4 * params.regMax + k];
                        if (rScore < maxScore)
                        {
                            maxScore = rScore;
                            maxInd = k;
                        }
                    }

                    maxScore *= scalar;

                    if (maxScore >= thresholdBeforeSigmoid)
                    {
                        maxScore = sigmoid(maxScore);

                        const MI_S16 *pBox = predsInt16;

                        for (int k = 0; k < 4; k++)
                        {
                            for (int i_reg = 0; i_reg < params.regMax; i_reg++)
                            {
                                dist_before_softmax[i_reg] = *(pBox + k * params.regMax + i_reg) * params.scalar[iLayer];
                            }

                            softmax_(dist_before_softmax, distribution, params.regMax);

                            ltrb[k] = 0.0;
                            for (int l = 0; l < params.regMax; l++)
                            {
                                ltrb[k] += l * distribution[l];
                            }
                            ltrb[k] = ltrb[k] * strideW;
                        }

                        MI_FLOAT pbCx = (j + 0.5) * strideW;
                        MI_FLOAT pbCy = (i + 0.5) * strideH;

                        BoxInfo_t rBoxInfo;
                        rBoxInfo.x1 = pbCx - ltrb[0];
                        rBoxInfo.x2 = pbCx + ltrb[2];
                        rBoxInfo.y1 = pbCy - ltrb[1];
                        rBoxInfo.y2 = pbCy + ltrb[3];
                        rBoxInfo.score = maxScore;
                        rBoxInfo.label = maxInd;
                        outBoxes.push_back(rBoxInfo);
                    }
                    predsInt16 += ALIGN_UP(numChannels, INNER_MOST_ALIGNMENT);
                }
            }
        }
        delete[] dist_before_softmax;
    }
    else
    {
        printf("Not implemented out format: %d !\n", params.outFormat);
        delete[] scores;
        delete[] distribution;
        exit(__LINE__);
    }

    delete[] scores;
    delete[] distribution;

    return MI_SUCCESS;
}

MI_S32 ST_ALGO_DetPureIpuPostProcessOneLayer(int iLayer, MI_IPU_TensorVector_t OutputTensorVector, std::vector<BoxInfo_t> &outBoxes)
{
    std::cout << "all layers: " << iLayer << std::endl;
    float *bboxes = (float *)(OutputTensorVector.astArrayTensors[0].ptTensorData[0]);
    short *class_id = (short *)(OutputTensorVector.astArrayTensors[1].ptTensorData[0]);
    float *scores = (float *)(OutputTensorVector.astArrayTensors[2].ptTensorData[0]);
    short *scores_index = (short *)(OutputTensorVector.astArrayTensors[3].ptTensorData[0]);
    short *box_num = (short *)(OutputTensorVector.astArrayTensors[4].ptTensorData[0]);
    int num_boxes = (int) (*(box_num));
    int all_boxes = 8400;

    for (int i = 0; i < num_boxes; i++)
    {
        int box_index = (int) (scores_index[i]);
        float p_box[4];
        for (int j=0; j<4; j++)
        {
            p_box[j] = *(bboxes + all_boxes*j + box_index);
        }

        int r_class = (int) (class_id == NULL ? 0 : class_id[box_index]);
        float r_score = scores[box_index];

        if (r_score >= 0.5)
        {
            BoxInfo_t rBoxInfo;
            rBoxInfo.x1 = p_box[0];
            rBoxInfo.y1 = p_box[1];
            rBoxInfo.x2 = p_box[2];
            rBoxInfo.y2 = p_box[3];
            rBoxInfo.score = r_score;;
            rBoxInfo.label = r_class;
            outBoxes.push_back(rBoxInfo);
        }
    }

    return MI_SUCCESS;
}


void ST_Algo_SaveResultImage(std::vector<BoxInfo_t> bboxes, std::string inPath, std::string outDir, cv::Mat &bufferMat)
{

    std::string imageBaseName = inPath.substr(inPath.rfind("/") + 1);

    size_t lastDotPos = imageBaseName.rfind(".");
    imageBaseName.replace(lastDotPos, imageBaseName.size() - lastDotPos, ".png");

    if (access(outDir.c_str(), F_OK) != 0)
    {
        mkdir(outDir.c_str(), 0777);
    }

    std::string outputImageDir = outDir;
    if (access(outputImageDir.c_str(), F_OK) != 0)
    {
        mkdir(outputImageDir.c_str(), 0777);
    }

    std::string outImagePath = outputImageDir + imageBaseName;

    for (size_t i = 0; i < bboxes.size(); i++)
    {
        if (bboxes[i].label >= (int) cocoClasses.size())
        {
            printf("invalid class id %d ignored!\n", bboxes[i].label);
            continue;
        }

        cv::Rect rect(
            bboxes[i].x1,
            bboxes[i].y1,
            bboxes[i].x2 - bboxes[i].x1,
            bboxes[i].y2 - bboxes[i].y1);

        int32_t rClassId = int32_t(bboxes[i].label);

        cv::Scalar rBoxColor = cv::Scalar(0, 0, 255);

        std::string rClassName(cocoClasses[rClassId]);

        cv::rectangle(bufferMat, rect, rBoxColor, 1, cv::LINE_8, 0);
        cv::putText(
            bufferMat,
            rClassName + "_0." + std::to_string(int(bboxes[i].score * 100)),
            cv::Point(bboxes[i].x1,
                      bboxes[i].y1),
            cv::FONT_HERSHEY_SIMPLEX,
            0.4,
            rBoxColor,
            1);
    }

    std::cout << "outImagePath: " << outImagePath << std::endl;
    cv::imwrite(outImagePath, bufferMat);
}

int main(int argc, char *argv[])
{
    MI_FLOAT threshold = 0.5;
    char modelPath[MAX_PATH_STRLEN] = {0};
    char pFirmwarePath[MAX_PATH_STRLEN] = "/config/dla/ipu_firmware.bin";
    MI_S32 s32Ret;

    struct timeval tstart, tend;

    std::string inImages;
    ST_Algo_ParseCmdOptions(argc, argv, inImages, modelPath, &threshold);

    AlgoInput_t algoInput; // buffer info for det model

    MI_U16 u16SocId = ST_DEFAULT_SOC_ID;
    MI_SYS_Init(u16SocId);

    MI_IPUChnAttr_t stChnAttr;
    MI_IPU_SubNet_InputOutputDesc_t desc;
    MI_IPU_OfflineModelStaticInfo_t OfflineModelInfo;
    memset(&stChnAttr, 0, sizeof(MI_IPUChnAttr_t));
    memset(&desc, 0, sizeof(MI_IPU_SubNet_InputOutputDesc_t));

    // 1. create device
    s32Ret = MI_IPU_GetOfflineModeStaticInfo(NULL, modelPath, &OfflineModelInfo);
    if (s32Ret != MI_SUCCESS)
    {
        printf("get model variable buffer size failed (%d)!\n", s32Ret);
        exit(__LINE__);
    }

    MI_IPU_DevAttr_t stDevAttr;
    stDevAttr.u32MaxVariableBufSize = OfflineModelInfo.u32VariableBufferSize;
    stDevAttr.u32YUV420_W_Pitch_Alignment = 16;
    stDevAttr.u32YUV420_H_Pitch_Alignment = 2;
    stDevAttr.u32XRGB_W_Pitch_Alignment = 16;
    s32Ret = MI_IPU_CreateDevice(&stDevAttr, NULL, pFirmwarePath, 0);

    if (s32Ret != MI_SUCCESS)
    {
        printf("create ipu device failed(%d)!\n", s32Ret);
        exit(__LINE__);
    }

    // 2. create channel
    stChnAttr.u32InputBufDepth = 0;
    stChnAttr.u32OutputBufDepth = 1;
    MI_U32 u32ChannelID;
    s32Ret = MI_IPU_CreateCHN(&(u32ChannelID), &stChnAttr, NULL, modelPath);
    if (s32Ret != MI_SUCCESS)
    {
        MI_IPU_DestroyDevice();
        printf("create detect ipu channel failed(%d)!\n", s32Ret);
        exit(__LINE__);
    }

    // 3. get input/output tensor
    MI_IPU_SubNet_InputOutputDesc_t modelDesc;

    s32Ret = MI_IPU_GetInOutTensorDesc(u32ChannelID, &(modelDesc));
    if (s32Ret != MI_SUCCESS)
    {
        MI_IPU_DestroyDevice();
        printf("get tensor desc failed(%d)!\n", s32Ret);
        exit(__LINE__);
    }

    // 4. init input buffer
    ModelAttr_t modelAttr;
    modelAttr.width = modelDesc.astMI_InputTensorDescs[0].u32TensorShape[2];
    modelAttr.height = modelDesc.astMI_InputTensorDescs[0].u32TensorShape[1];
    modelAttr.format = modelDesc.astMI_InputTensorDescs[0].eElmFormat;
    ST_Algo_InitBuffer(u16SocId, &modelAttr, &algoInput);

    cv::Mat bufferMat(modelAttr.height, modelAttr.width, CV_8UC3);

    std::vector<cv::String> inputPaths;

    cv::glob(cv::String(inImages.c_str()), inputPaths, false);
    std::cout << "found " << inputPaths.size() << " images!" << std::endl;

    std::string time_str = std::to_string(long(time(0)));

    std::string outDir = "./output/" + time_str + "/";
    // if (access(outDir.c_str(), F_OK) != 0)
    // {
    //     mkdir(outDir.c_str(), 0777);
    // }
    ST_Algo_Create_Directory(outDir);

    for (size_t i = 0; i < inputPaths.size(); i++)
    {

        std::cout << "[" << i << "] "
                  << "processing " << inputPaths[i] << "..." << std::endl;

        // 5. fill input buffer

        ST_Algo_FillBuffer(inputPaths[i], &modelAttr, &algoInput);
        // dump input buffer into an opencv mat
        ST_Algo_DumpBuffer(&modelAttr, &algoInput, bufferMat);

        gettimeofday(&tstart, NULL);

        // 6. invoke model
        MI_IPU_TensorVector_t InputTensorVector;
        MI_IPU_TensorVector_t OutputTensorVector;
        memset(&InputTensorVector, 0, sizeof(MI_IPU_TensorVector_t));
        InputTensorVector.u32TensorCount = modelDesc.u32InputTensorCount;
        s32Ret = MI_IPU_GetOutputTensors(u32ChannelID, &OutputTensorVector);
        if (s32Ret != MI_SUCCESS)
        {
            printf("get output tensor failed,ret:%d\n", s32Ret);
            exit(__LINE__);
        }

        InputTensorVector.astArrayTensors[0].phyTensorAddr[0] = algoInput.phyTensorAddr;
        InputTensorVector.astArrayTensors[0].ptTensorData[0] = algoInput.ptTensorData;

        if (MI_SUCCESS != (s32Ret = MI_IPU_Invoke(u32ChannelID, &InputTensorVector, &OutputTensorVector)))
        {
            printf("MI_IPU_Invoke error, ret %d", s32Ret);
            MI_IPU_PutOutputTensors(u32ChannelID, &OutputTensorVector);
            MI_IPU_PutInputTensors(u32ChannelID, &InputTensorVector);
            MI_IPU_DestroyCHN(u32ChannelID);
            MI_IPU_DestroyDevice();
            exit(__LINE__);
        }

        s32Ret = MI_IPU_PutOutputTensors(u32ChannelID, &OutputTensorVector);
        if (s32Ret != MI_SUCCESS)
        {
            printf("failed to put output tensor, return %d\n", s32Ret);
            exit(__LINE__);
        }

        gettimeofday(&tend, NULL);

        printf("model invoke time: %f ms\n",
               (tend.tv_sec - tstart.tv_sec) * 1000.0 + (tend.tv_usec - tstart.tv_usec) / 1000.0);

        tstart = tend;
        // 7. output decode
        DetParams_t params;

        if (modelDesc.u32OutputTensorCount > MAX_NUM_OUT_TENSOR)
        {
            printf(
                "num output tensor channel should not be larger than %d\n",
                MAX_NUM_OUT_TENSOR);
            exit(__LINE__);
        }

        params.netInW = modelAttr.width;
        params.netInH = modelAttr.height;
        params.outFormat = modelDesc.astMI_OutputTensorDescs[0].eElmFormat;;
        params.nmsThreshold = 0.5;
        params.confThreshold = threshold;

        // printf("params.numClasses = %d\n",  params.numClasses);

        params.numOutLayers = modelDesc.u32OutputTensorCount;

        for (MI_U32 i = 0; i < params.numOutLayers && i < MAX_NUM_OUT_TENSOR; i++)
        {
            params.data[i] = (void *)(OutputTensorVector.astArrayTensors[i].ptTensorData[0]);

            memcpy(params.outShapes[i], modelDesc.astMI_OutputTensorDescs[i].u32TensorShape, sizeof(params.outShapes[i]));
            params.scalar[i] = modelDesc.astMI_OutputTensorDescs[i].fScalar;

            // printf("layer %d shape = %u  x %u x %u x %u \n",
            //     i,
            //     params.outShapes[i][0],
            //     params.outShapes[i][1],
            //     params.outShapes[i][2],
            //     params.outShapes[i][3]);
        }


        std::vector<BoxInfo_t> outBoxes;

        if (params.numOutLayers == 3) // deocode in cpu
        {
            for (MI_U32 iLayer = 0; iLayer < params.numOutLayers && iLayer < MAX_NUM_OUT_TENSOR; iLayer++)
            {
                struct timeval tstartDecode, tendDecode;
                gettimeofday(&tstartDecode, NULL);
                ST_ALGO_DetCpuPostProcessOneLayer(
                    params,
                    iLayer,
                    outBoxes);
                gettimeofday(&tendDecode, NULL);
                // printf("decoding layer %d time: %f ms\n", iLayer,
                //        (tendDecode.tv_sec - tstartDecode.tv_sec) * 1000.0 + (tendDecode.tv_usec - tstartDecode.tv_usec) / 1000.0);
            }
        }
        else if (params.numOutLayers == 1) //  decode in ipu
        {
            struct timeval tstartDecode, tendDecode;
            gettimeofday(&tstartDecode, NULL);
            ST_ALGO_DetIpuPostProcessOneLayer(
                params,
                0,
                outBoxes);
            gettimeofday(&tendDecode, NULL);
            // printf("decoding layer %d time: %f ms\n", iLayer,
            //        (tendDecode.tv_sec - tstartDecode.tv_sec) * 1000.0 + (tendDecode.tv_usec - tstartDecode.tv_usec) / 1000.0);
        }
        else if(params.numOutLayers == 5)
        {
            ST_ALGO_DetPureIpuPostProcessOneLayer(
                params.numOutLayers,
                OutputTensorVector,
                outBoxes);
        }
        else
        {
            printf("unexpected num output tensors(%d)", params.numOutLayers);
            exit(__LINE__);
        }

        ST_ALGO_CheckBoxes(outBoxes, params.netInW, params.netInH);
        struct timeval tstartNms, tendNms;
        gettimeofday(&tstartNms, NULL);
        ST_ALGO_Nms(outBoxes, params.nmsThreshold);
        gettimeofday(&tendNms, NULL);
        // printf("Nms time: %f ms\n",
        //        (tendNms.tv_sec - tstartNms.tv_sec) * 1000.0 + (tendNms.tv_usec - tstartNms.tv_usec) / 1000.0);

        gettimeofday(&tend, NULL);

        printf("post process time: %f ms\n",
               (tend.tv_sec - tstart.tv_sec) * 1000.0 + (tend.tv_usec - tstart.tv_usec) / 1000.0);

        ST_Algo_SaveResultImage(outBoxes, inputPaths[i], outDir, bufferMat);
    }

    MI_SYS_FlushInvCache(algoInput.ptTensorData, algoInput.u32BufSize);
    MI_SYS_Munmap(algoInput.ptTensorData, algoInput.u32BufSize);
    MI_SYS_MMA_Free(u16SocId, algoInput.phyTensorAddr);
    MI_IPU_DestroyCHN(u32ChannelID);
    MI_IPU_DestroyDevice();
    MI_SYS_Exit(u16SocId);
}