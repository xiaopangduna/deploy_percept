# deploy_percept

## 项目简介

边缘 AI 推理框架，提供 YOLO 系列目标检测 / 实例分割 / 姿态估计，以及分类模型的推理与后处理。采用模块化 CMake 工程，支持宿主机开发验证与交叉编译部署。

### 平台与 CMake Preset

| Preset | 用途 | NPU 引擎 |
|--------|------|----------|
| `x86_64-debug` | 宿主机开发、后处理单测 | — |
| `aarch64-linux-gnu_orange_pi_4_pro_a733-release` | Orange Pi A733 交叉编译与板端部署 | AWNN |
| `armv7l-SSC375-release` | Sigmastar 预留 | 未启用 |

开发容器：`.devcontainer/ubuntu-22.04-allwinner/`（详见 [.devcontainer/README.md](.devcontainer/README.md)）。

## 构建与测试

### 1. 工具链

- **x86_64**：无需额外配置，使用宿主机或 devcontainer 自带编译器。
- **aarch64-linux-gnu_orange_pi_4_pro_a733**：下载 `gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu`，放置于 `/opt/toolchains/`
  - 下载：https://mirrors.tuna.tsinghua.edu.cn/armbian-releases/_toolchain/

### 2. 第三方库

构建产物安装至 `third_party/<platform>/`（`<platform>` 与 preset 对应 toolchain 中的平台 tag 一致）。

- **公开库**（cnpy、gtest、opencv、spdlog 等）：由 `scripts/third_party_builder.sh` 调度，各库的下载与编译脚本在 [`scripts/third_party_builders/`](scripts/third_party_builders/)（`builder_*.sh`），可直接构建。
- **非公开 / 平台专属库**（如 AWNN、RKNN、RGA）：不在上述脚本内自动下载，需按 [`cmake/modules/`](cmake/modules/) 中对应 `Find*.cmake` 的说明手动准备（例如 AWNN 见 `FindAwnn.cmake`，VIPLite 放入 `third_party/<platform>/awnn/`）。

```bash
bash scripts/third_party_builder.sh --help
bash scripts/third_party_builder.sh <platform> --libs <libs>
```

```bash
# x86_64
bash scripts/third_party_builder.sh x86_64 --libs all

# aarch64-linux-gnu_orange_pi_4_pro_a733
bash scripts/third_party_builder.sh aarch64-linux-gnu_orange_pi_4_pro_a733 --libs cnpy,gtest,opencv,spdlog
```

### 3. 构建

```bash
bash scripts/build.sh --preset <preset>
```

```bash
bash scripts/build.sh --preset x86_64-debug
bash scripts/build.sh --preset aarch64-linux-gnu_orange_pi_4_pro_a733-release
```

### 4. 安装

```bash
bash scripts/install.sh --preset <preset>
```

```bash
bash scripts/install.sh --preset x86_64-debug
bash scripts/install.sh --preset aarch64-linux-gnu_orange_pi_4_pro_a733-release
```

### 5. 测试

```bash
# x86_64：本地
bash scripts/test.sh --preset x86_64-debug

# aarch64-linux-gnu_orange_pi_4_pro_a733：远程

# 需先设置ssh密码
export SSHPASS='your_password'

bash scripts/test.sh --preset aarch64-linux-gnu_orange_pi_4_pro_a733-release \
  --board orangepi@192.168.0.103:~/deploy_percept

# 同步
rsync -avz --delete -e ssh \
  install/aarch64-linux-gnu_orange_pi_4_pro_a733/ \
  orangepi@192.168.0.103:~/deploy_percept/
```

## 架构说明

### 目录结构

```
deploy_percept/
├── CMakeLists.txt
├── CMakePresets.json              # configure / build preset
├── README.md
├── apps/                          # 见示例应用
├── benchmarks/                    # 性能 benchmark（AWNN，ENABLE_BENCHMARKS）
├── cmake/
│   ├── FindThirdParty.cmake       # 第三方库查找唯一入口
│   ├── PerceptInstall.cmake
│   ├── modules/                   # Find*.cmake
│   └── toolchains/
├── include/deploy_percept/
│   ├── common/                    # types 等公共契约
│   ├── core/                      # engine、post_process、deploy_percept.hpp
│   ├── data/                      # COCO / ImageNet 标签
│   └── utils/                     # 预处理、可视化、io、fixture_path
├── scripts/
│   ├── build.sh                   # 构建脚本
│   ├── install.sh                 # 安装脚本
│   ├── test.sh                    # 测试脚本
│   ├── third_party_builder.sh     # 多个第三方库安装调度脚本
│   └── third_party_builders/      # 单个第三方库安装调度脚本
├── src/
│   ├── common/
│   ├── core/                      # deploy_percept_core
│   └── utils/                     # deploy_percept_utils
├── tests/
│   ├── fixtures/                  # install → share/percept/fixtures/
│   ├── smoke/
│   ├── unit/post_process/
│   ├── integration/awnn/
│   └── test_common/
└── third_party/                   # 第三方预编译产物
```

### 库模块

| 模块 | CMake target | 头文件入口 | 说明 |
|------|--------------|------------|------|
| **common** | `deploy_percept_common` | `common/types.hpp` | 类型与公共契约，INTERFACE |
| **core** | `deploy_percept_core` | `core/deploy_percept.hpp` + `engine/`、`post_process/` | 推理与后处理，不依赖 OpenCV |
| **utils** | `deploy_percept_utils` | `utils/deploy_percept_utils.hpp` | cnpy + OpenCV + spdlog 齐备才编译；RGA 按平台可选 |
| **apps** | — | 按需 include | 通常链 `core` + `utils` |
| **tests** | 分层门禁见下 | — | unit / integration 依赖 `deploy_percept_utils` |

需要推理链 `deploy_percept_core`；需要 vis / pre_process / io 等再链 `deploy_percept_utils`（demo / 单测通常两者都链）。

### 依赖项

- **OpenCV / spdlog / cnpy**：齐备后编译 `deploy_percept_utils`（core 不依赖 OpenCV）
- **GTest**：测试
- **AWNN / VIPLite**：A733 平台 NPU 运行时（需手动放入 `third_party/.../awnn/viplite-tina`，见 [示例应用](#示例应用) 各 app README）

第三方库查找：`FindThirdParty.cmake` 为唯一入口。

## 示例应用

示例位于 [`apps/`](apps/)，按芯片与任务索引；模型准备与运行步骤见各应用文档。

| 芯片 | 任务 | 应用 | 文档 |
|------|------|------|------|
| A733 | 目标检测 | `yolov5_detect_awnn` | [README](apps/yolov5_detect_awnn/README.md) |
| A733 | 目标检测 | `yolov8_detect_awnn` | [README](apps/yolov8_detect_awnn/README.md) |
| A733 | 图像分类 | `mobilenetv2_classify_awnn` | [README](apps/mobilenetv2_classify_awnn/README.md) |
| RK3588 | 目标检测 | `yolov5_detect_rknn` | — |
| RK3588 | 实例分割 | `yolov5_seg_rknn` | — |
| RK3588 | 实例分割 | `yolov8_seg_rknn` | [README](apps/yolov8_seg_rknn/README.md) |
| RK3588 | 姿态估计 | `yolov8_pose_rknn` | — |

