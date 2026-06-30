# 概述

本文档描述yolov8模型在NPU的部署过程，含模型转换与板端示例两部分，其中`convert_model`为模型转换的目录，其它文件为板端运行的示例代码等文件。



# YOLOv8

YOLOv8 是 ultralytics 公司在 2023 年 1月 10 号开源的 YOLOv5 的下一个重大更新版本，目前支持图像分类、物体检测和实例分割任务，受到用户的广泛关注。

![banner-yolo-vision-2023](figures/banner-yolo-vision-2023.png)

按照官方描述，YOLOv8 是一个 SOTA 模型，它建立在以前 YOLO 版本的成功基础上，并引入了新的功能和改进，以进一步提升性能和灵活性。具体创新包括一个新的骨干网络、一个新的 Ancher-Free 检测头和一个新的损失函数，可以在从 CPU 到 GPU 的各种硬件平台上运行。

![yolo-comparison-plots](figures/yolo-comparison-plots.png)

本次ultralytics 并没有直接将开源库命名为 YOLOv8，而是直接使用 ultralytics 这个词，原因是 ultralytics 将这个库定位为算法框架，而非某一个特定算法，一个主要特点是可扩展性。



# 模型获取

## 获取源码

git clone -b v8.1.0 https://github.com/ultralytics/ultralytics.git



## 下载原始模型

下载yolov8n.pt模型文件

https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt



## 环境要求

Pip install the ultralytics package including all [requirements](https://github.com/ultralytics/ultralytics/blob/main/pyproject.toml) in a [**Python>=3.8**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).

安装ultralytics软件包：

```python
pip install ultralytics==8.1.0
```



## 转onnx格式

将torch 格式模型转为onnx格式模型：

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.export(format='onnx', dynamic=True, opset=11)
```

## **固定shape**

```python
python3 -m onnxsim yolov8n.onnx yolov8n_640_sim.onnx --input-shape=1,3,640,640
```



## **模型测试**

```python
yolo predict model=yolov8n_640_sim.onnx source=ultralytics/assets/bus.jpg
```

![bus](figures/bus.jpg)

验证转出的onnx模型文件正常推理。

## 模型修改

yolov8网络的后处理部分（输出head节点往后的layer）8bit量化会产生较大的精度损失， 裁剪网络后处理节点，后处理的计算跑CPU。

```bash
cd ./convert_model/python/

python3 onnx_extract.py
```



# 模型转换

模型转换主要包含原始模型导入、量化、导出为NPU可识别的模型格式等步骤。

```bash
cd ./convert_model/
```

修改config_yml.py文件的相关参数配置；

```python
# "database"
DATASET = ['../../dataset/coco_12/dataset.txt']
DATASET_TYPE = ["TEXT"]

# mean, scale
MEAN    = [0, 0, 0]
SCALE   = [0.0039216, 0.0039216, 0.0039216]

# reverse_channel: True bgr, False rgb
REVERSE_CHANNEL = False

# add_preproc_node, True or False
ADD_PREPROC_NODE = True
# "preproc_type"
PREPROC_TYPE = ["IMAGE_RGB"]

# add_postproc_node, quant output -> float32 output
ADD_POSTPROC_NODE = True
```

模型导入、量化、导出等步骤：

```bash
# using xxx_env.sh to create softlink
./convert_model_env.sh

# 导入
# pegasus_import.sh <model_name>
./pegasus_import.sh yolov8n_6

# 量化
# pegasus_quantize.sh <model_name> <quantize_type> <calibration_set_size>
./pegasus_quantize.sh yolov8n_6 uint8 12

# 仿真（可选）
# pegasus_inference.sh <model_name> <quantize_type>
./pegasus_inference.sh yolov8n_6 uint8

# 导出nb模型
# pegasus_export_ovx_nbg.sh <model_name> <quantize_type> <platform>
./pegasus_export_ovx_nbg.sh yolov8n_6 uint8 t527
# 导出的模型文件存放在../model目录
# 例如 ../model/yolov8n_6_uint8_t527.nb
```



# 板端demo

含demo编译及运行说明。

## 解压opencv压缩包

```bash
# 进入目录
cd ../../../3rdparty/opencv/
# 解压，选择对应平台
# armhf, eg: V85x, R853
unzip opencv-3.4.16-gnueabihf-linux.zip
# linux aarch64, eg: T527/MR527/MR536/T536/A733/T736
unzip opencv-4.9.0-aarch64-linux-sunxi-glibc.zip
# android aarch64, eg: T527/A733/T736
unzip opencv-4.9.0-android.zip
```



## 准备交叉编译工具链

### Linux

```bash
# 进入目录
cd ../../0-toolchains/
# 解压
# armhf, V85x, R853
unzip arm-openwrt-linux-muslgnueabi.zip
chmod 777 -R ./arm-openwrt-linux-muslgnueabi
# aarch64, MR527, T527, MR536, T536, A733, T736
tar xvf gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu.tar.xz
# aarch64 for debian11, T527, A733, T736
tar vxf gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu.tar.xz
```

编译脚本会根据平台自动选择交叉编译工具链，若需使用其它路径的工具链，可在`cmake_toolchain`目录修改`.cmake`文件内容指定对应的交叉编译工具链路径。



### Android

下载Android NDK，下载地址：https://developer.android.google.cn/ndk/downloads?hl=zh-cn

将下载的NDK放到编译机器目录，例如：./0-toolchains/ ;

请根据下载的版本修改`cmake_toolchain`目录的`android_ndk_build_env.sh` 文件。

使用unzip命令解压。



## build && run

### Linux

在Linux系统下测试。编译用法如下：

```bash
# 途径一：在yolov8目录编译
cd ../examples/yolov8/
./../build_linux.sh -t <platform> [-s <system>]
# 途径二：在examples目录，再选择yolov8目录编译
cd ../examples
./build_linux.sh -t <platform> -p yolov8 [-s <system>]
```

以下说明以T527平台为例；

```bash
cd ../examples/yolov8/
./../build_linux.sh -t t527
```

> 若是T527平台debian系统，则是以下命令：
>
> ```bash
> cd ../examples/yolov8/
> ./../build_linux.sh -t t527 -s debian11
> ```

push 可执行文件、模型文件、输入图片到板端目录（建议推到tf卡目录，空间充足）；

```
adb push install\yolov8_demo_linux_t527 /mnt/UDISK/
```

运行；

```bash
adb shell
cd /mnt/UDISK/yolov8_demo_linux_t527

# 可选
export LD_LIBRARY_PATH=./lib

# 运行可执行文件
# ./yolov8_demo_t527 -h 查看执行示例说明
chmod +x ./yolov8_demo_t527
./yolov8_demo_t527 -nb model/yolov8n_6_uint8_t527.nb -i model/dog.jpg
```

运行后，打印log输出，能看到检测信息输出，并将检测结果画框保存为图片out_yolov8.png。

![out_yolov8_640](figures/out_yolov8_640.png)



### Android

在Android 64bit系统下测试。编译用法如下：

```bash
# 途径一：在yolov8目录编译
cd ../examples/yolov8/
./../build_android.sh -t <platform>
# 途径二：在examples目录，再选择yolov8目录编译
cd ../examples
./build_android.sh -t <platform> -p yolov8
```

以下说明以T527平台为例；

```bash
cd ../examples/yolov8/
./../build_android.sh -t t527
```

修改权限；

```bash
adb root
adb remount
```

push 可执行文件、模型文件、输入图片到`/data/local/`目录；

```bash
adb push install\yolov8_demo_android_t527 /data/local/
```

运行；

```bash
adb shell
cd /data/local/yolov8_demo_android_t527

export LD_LIBRARY_PATH=./lib

# 运行可执行文件
chmod +x ./yolov8_demo_t527
./yolov8_demo_t527 -nb model/yolov8n_6_uint8_t527.nb -i model/dog.jpg
```

运行后，打印log输出，能看到检测信息输出。
