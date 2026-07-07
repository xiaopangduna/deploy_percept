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

- **公开库**（zlib、cnpy、gtest、opencv、spdlog 等）：由 `scripts/third_party_builder.sh` 调度，各库的下载与编译脚本在 [`scripts/third_party_builders/`](scripts/third_party_builders/)（`builder_*.sh`），可直接构建。
- **`--link static|shared`**：控制构建产物为静态库或动态库，**默认 static**。构建前会清除该库在 `third_party/<platform>/<lib>/` 下的旧安装，避免 `.a`/`.so` 并存（**zlib、cnpy 例外**：上游默认同时安装静/动态库；主工程分别通过 CMake 选项 `ZLIB_LINK_TYPE`、`CNPY_LINK_TYPE` 选择链哪一个，**默认均为 STATIC**）。
- **混链**：third_party 只决定各库安装形态；与 AWNN 等 `.so` 或其它库的静/动组合，由主工程 CMake 自行配置。
- **非公开 / 平台专属库**（如 AWNN、RKNN、RGA）：不在上述脚本内自动下载，需按 [`cmake/modules/`](cmake/modules/) 中对应 `Find*.cmake` 的说明手动准备（例如 AWNN 见 `FindAwnn.cmake`，将 `awnpu_model_zoo` 的 `common/npuruntime` 整包置于 `third_party/<platform>/awnn/npuruntime`）。rga/rknpu 仅提供 shared，请求 `--link static` 时会警告并仍安装 `.so`。

```bash
bash scripts/third_party_builder.sh --help
bash scripts/third_party_builder.sh <platform> --libs <libs> [--link static|shared]
```

```bash
# x86_64（默认 static）
bash scripts/third_party_builder.sh x86_64 --libs zlib,opencv,spdlog

# shared 动态库（开发机；更换 third_party 产物后需 rm -rf build/<preset> 再 configure）
bash scripts/third_party_builder.sh x86_64 --libs zlib,opencv --link shared

# aarch64-linux-gnu_orange_pi_4_pro_a733
bash scripts/third_party_builder.sh aarch64-linux-gnu_orange_pi_4_pro_a733 --libs cnpy,zlib,gtest,opencv,spdlog
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

install 包布局（`install/<platform>/`）：

```
install/<platform>/
├── bin/                          # demo 可执行文件
├── lib/                          # 静态库、VIPLite .so
├── include/deploy_percept/       # 对外头文件
├── tests/                        # 测试可执行文件（INSTALL_TESTS）
├── benchmarks/                   # benchmark（ENABLE/INSTALL_BENCHMARKS）
├── scripts/run_install_tests.sh  # install 包测试 runner
└── share/percept/fixtures/       # 测试与 demo 共用数据
```

### 5. 测试

**PC 全流程**（build → install → 跑 install 包）：

```bash
bash scripts/test.sh --preset x86_64-debug
```

**开发板**（build → install → rsync → 远程跑测）：

```bash
export SSHPASS='your_password'   # 或配置 ssh-copy-id 密钥

bash scripts/test.sh --preset aarch64-linux-gnu_orange_pi_4_pro_a733-release \
  --board orangepi@192.168.0.103:~/deploy_percept

# 含 benchmark（默认 50 loops）
bash scripts/test.sh --preset aarch64-linux-gnu_orange_pi_4_pro_a733-release \
  --board orangepi@192.168.0.103:~/deploy_percept --bench 100
```

**install 包内直接跑**（PC 或开发板，rsync 后）：

```bash
cd ~/deploy_percept
./scripts/run_install_tests.sh              # 扫描 tests/ 下全部可执行文件
./scripts/run_install_tests.sh --bench 50   # tests + benchmarks
```

手动同步 install 包：

```bash
rsync -avz --delete -e ssh \
  install/aarch64-linux-gnu_orange_pi_4_pro_a733/ \
  orangepi@192.168.0.103:~/deploy_percept/
```

开发期 build tree 单测：`cd build/<preset> && ctest --output-on-failure`

## 架构说明

### 目录结构

```
deploy_percept/
├── CMakeLists.txt
├── CMakePresets.json              # configure / build preset
├── README.md
├── apps/                          # 示例应用
├── benchmarks/                    # 性能 benchmark（AWNN，ENABLE_BENCHMARKS）
├── cmake/
│   ├── FindThirdParty.cmake       # 第三方库查找唯一入口
│   ├── modules/                   # Find*.cmake
│   └── toolchains/
├── include/deploy_percept/
│   ├── common/                    # types 等公共契约
│   ├── core/                      # engine、post_process、deploy_percept.hpp
│   ├── data/                      # COCO / ImageNet 标签
│   └── utils/                     # 预处理、可视化、io、fixture_path
├── scripts/
│   ├── build.sh                   # 构建
│   ├── install.sh                 # 安装
│   ├── test.sh                    # build + install + 验证 install 包
│   ├── run_install_tests.sh       # install 包内测试 runner（随包安装）
│   ├── third_party_builder.sh
│   └── third_party_builders/
├── src/
│   ├── common/                    # deploy_percept_common（INTERFACE）
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

- **OpenCV / spdlog / cnpy**：齐备后编译 `deploy_percept_utils`（core 不依赖 OpenCV；Find 模块均为非致命，缺失时跳过 utils）
- **GTest**：测试
- **AWNN / VIPLite**：A733 平台 NPU 运行时（见 `FindAwnn.cmake`，`third_party/<platform>/awnn/npuruntime`）

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
