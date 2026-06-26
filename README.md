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
├── cmake/                         # CMake 配置
│   ├── toolchains/                # 交叉编译工具链
│   └── modules/                   # Find*.cmake 第三方库查找模块
├── examples/                      # 示例数据
│   └── data/
├── include/                       # 公共头文件
│   └── deploy_percept/
│       ├── engine/                # 推理引擎接口
│       ├── post_process/          # 后处理算法
│       ├── pre_process/           # 预处理算法
│       ├── utils/                 # 工具函数
│       └── deploy_percept.hpp     # 主要入口头文件
├── scripts/                       # 构建与测试脚本
│   ├── build.sh                   # 编译（cmake configure + build）
│   ├── install.sh                 # 安装（cmake install → install/<platform>/）
│   ├── test.sh                    # 测试（ctest / install 包 / 开发板 SSH）
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

## 编译与测试

### 脚本职责

| 脚本 | 作用 |
|------|------|
| `scripts/build.sh` | 仅编译，产物在 `build/<preset>/` |
| `scripts/install.sh` | 仅安装，打包到 `install/<platform>/`（需先 build） |
| `scripts/test.sh` | 仅测试：build tree `ctest`、本地 install 包、开发板 SSH |

### 环境准备

在 devcontainer 中打开项目。Orange Pi 4 Pro (A733) 交叉工具链：

```bash
# 工具链: gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu
# 挂载至容器 /opt/toolchains
# 参考: https://mirrors.tuna.tsinghua.edu.cn/armbian-releases/_toolchain/
```

安装第三方库：

```bash
# 宿主机 x86_64
bash scripts/third_party_builder.sh x86_64 --libs all

# Orange Pi A733 交叉编译
bash scripts/third_party_builder.sh aarch64-linux-gnu_orange_pi_4_pro_a733 --libs cnpy,gtest,opencv,spdlog,yaml-cpp
```

### 宿主机开发（x86_64）

```bash
bash scripts/build.sh --preset x86_64-debug
bash scripts/test.sh --preset x86_64-debug
```

### 交叉编译（Orange Pi 4 Pro A733）

```bash
bash scripts/build.sh --preset aarch64-linux-gnu_orange_pi_4_pro_a733-release
bash scripts/install.sh --preset aarch64-linux-gnu_orange_pi_4_pro_a733-release
```

也可直接使用 CMake preset：

```bash
cmake --preset=aarch64-linux-gnu_orange_pi_4_pro_a733-release
cmake --build --preset=aarch64-linux-gnu_orange_pi_4_pro_a733-release
cmake --install build/aarch64-linux-gnu_orange_pi_4_pro_a733-release
```

### 测试

#### 1. build tree（宿主机日常开发）

```bash
bash scripts/build.sh --preset x86_64-debug
bash scripts/test.sh --preset x86_64-debug
```

等价于在 `build/x86_64-debug/` 下执行 `ctest --output-on-failure`。

#### 2. install tree（验证安装包）

```bash
bash scripts/build.sh --preset x86_64-debug
bash scripts/install.sh --preset x86_64-debug
bash scripts/test.sh --install-dir install/x86_64
```

#### 3. 开发板（交叉编译产物上板验证）

将 install 包同步到板子，在 PC 上通过 SSH 远程触发测试（板子无需 cmake/ctest/源码）：

```bash
# 1. 交叉编译 + 打 install 包
bash scripts/build.sh --preset aarch64-linux-gnu_orange_pi_4_pro_a733-release
bash scripts/install.sh --preset aarch64-linux-gnu_orange_pi_4_pro_a733-release

# 2. 拷贝 install 目录到开发板
rsync -avz install/aarch64-linux-gnu_orange_pi_4_pro_a733/ \
  orangepi@192.168.0.103:~/deploy_percept/

# 3. PC 上远程跑测（输出回显到 PC 终端）
bash scripts/test.sh --board orangepi@192.168.0.103:~/deploy_percept
```

**PC 依赖**：`ssh` 客户端，能登录板子（密码或 SSH 密钥）。

**板子依赖**：完整 install 目录（`bin/`、`share/percept/apps/`、`share/percept/tests/`），以及系统基础库（`libc`、`libstdc++`）。无需安装 CMake 或拷贝源码树。

路径可使用 `~/deploy_percept` 或绝对路径 `/home/orangepi/deploy_percept`。

**预期结果**（已安装项均 `[  PASSED  ]`；未安装或平台不适用则 skip）：

- `share/percept/tests/smoke_tests`
- `share/percept/tests/test_YoloV5DetectPostProcess`
- `share/percept/tests/test_YoloV5SegPostProcess`（mask 比对允许约 3% 像素容差，板子上可能打印 `Vectors are not equal` 仍可通过）
- `share/percept/tests/test_YoloV8SegPostProcess`
- `share/percept/tests/test_yolov5_detect_awnn`（仅 AWNN/aarch64 install 包）

构建 install 包时需 `ENABLE_TESTS=ON`（默认）且 `INSTALL_TESTS=ON`（默认）。量产包可 `-DINSTALL_TESTS=OFF`，仅保留 `bin/` demo。

### install 目录结构

```
install/<platform>/
├── bin/                         # demo 可执行文件（如 yolov5_detect_awnn）
├── lib/                         # libdeploy_percept.a、VIPLite 等
├── include/deploy_percept/      # 头文件
├── share/percept/
│   ├── apps/                    # 示例/测试 fixture 数据
│   └── tests/                   # 测试可执行文件（INSTALL_TESTS=ON）
└── var/percept/output/          # 测试输出（运行时自动创建）
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