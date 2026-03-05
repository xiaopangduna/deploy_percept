# deploy_percept

一个边缘AI推理框架，支持YOLO系列模型的目标检测和实例分割。

## 项目简介

`deploy_percept` 是一个高性能边缘AI推理框架。该项目提供了一套完整的解决方案，支持YOLOv5、YOLOv8等主流模型的目标检测和实例分割任务。框架采用模块化设计，便于扩展和维护。

## 功能特性

- **多模型支持**：支持YOLOv5、YOLOv8等主流目标检测和分割模型

## 项目结构

```
deploy_percept/
├── CMakeLists.txt                 # 根CMakeLists.txt，配置整个项目
├── README.md
├── apps/                          # 示例应用
│   ├── yolov5_detect_rknn/        # YOLOv5目标检测示例
│   ├── yolov5_seg_rknn/           # YOLOv5实例分割示例
│   └── yolov8_seg_rknn/           # YOLOv8实例分割示例
├── cmake/                         # CMake模块
│   ├── Find*.cmake                # 第三方库查找脚本
│   └── aarch64-toolchain.cmake    # ARM交叉编译工具链
├── examples/                      # 示例数据
│   └── data/
├── include/                       # 公共头文件
│   └── deploy_percept/
│       ├── engine/                # 推理引擎接口
│       ├── post_process/          # 后处理算法
│       ├── pre_process/           # 预处理算法
│       ├── utils/                 # 工具函数
│       └── deploy_percept.hpp     # 主要入口头文件
├── scripts/                       # 构建脚本
│   ├── build.sh                   # 构建脚本
│   ├── test.sh                    # 测试脚本
│   └── third_party_builders/      # 第三方库构建脚本
├── src/                           # 源代码
│   ├── engine/                    # 推理引擎实现
│   ├── post_process/              # 后处理算法实现
│   ├── pre_process/               # 预处理算法实现
│   └── utils/                     # 工具函数实现
├── tests/                         # 测试代码
│   ├── post_process/              # 后处理算法测试
│   ├── test_common/               # 测试通用组件
│   └── utils/                     # 测试工具函数
└── third_party/                   # 第三方依赖（可能在.gitignore中）
```

## 依赖项

- **RKNN Toolkit**：瑞芯微神经网络推理工具包
- **OpenCV**：计算机视觉库
- **CMake**：构建系统
- **spdlog**：日志库
- **yaml-cpp**：YAML解析库
- **cnpy**：NumPy数组读写库

## 编译步骤

### 环境准备

安装第三方库：

```bash
# aarch64
bash scripts/third_party_builder.sh aarch64 --libs all
# x86_64
bash scripts/third_party_builder.sh x86_64 --libs all

bash scripts/third_party_builder.sh aarch64 --libs cnpy,gtest,opencv,rga,rknpu,spdlog,yaml-cpp
```

### 标准构建

```bash
# 配置
cmake --preset=x86_64-debug-host
# 构建
cmake --build --preset=x86_64-debug-host
# 安装
cmake --install build/x86_64-debug-host
--preset=x86_64-debug-host/aarch64-release-cross/aarch64-debug-host
```

### 交叉编译（针对ARM平台）

```bash
cmake --preset=aarch64-release-cross
cmake --build --preset=aarch64-release-cross
```

### 测试
```bash
cd build/x86_64-debug-host && ctest
```

## 使用示例

项目提供了几个使用示例：
### rknn
1. **yolov5_detect_rknn**：YOLOv5目标检测示例
2. **yolov5_seg_rknn**：YOLOv5实例分割示例
3. **yolov8_seg_rknn**：YOLOv8实例分割示例

## 性能优化

- 使用RKNN硬件加速推理
- 优化内存分配和数据传输
- 采用高效的后处理算法

## 许可证

本项目采用 MIT 许可证，请参阅 LICENSE 文件获取更多信息。