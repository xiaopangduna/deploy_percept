# deploy_percept

## 项目简介

边缘 AI 推理框架，提供 YOLO 系列目标检测 / 实例分割 / 姿态估计，以及分类模型的推理与后处理。采用模块化 CMake 工程，支持宿主机开发验证与交叉编译部署。

### 平台与 CMake Preset

| Preset | 用途 | NPU 引擎 |
|--------|------|----------|
| `x86_64-debug` | 宿主机开发、后处理单测 | — |
| `aarch64-linux-gnu_orange_pi_4_pro_a733-release` | Orange Pi A733 交叉编译与板端部署 | AWNN |
| `armv7l-linux-gnueabihf_comake_pi_d1_ssd2355-release` | CoMake Pi D1 (SSD2355) 交叉编译 | Sigmastar IPU（`yolov8_detect_sigmastar`） |

开发容器：`.devcontainer/ubuntu-22.04-allwinner/`、`.devcontainer/ubuntu-20.04-sigmastar/`（详见 [.devcontainer/README.md](.devcontainer/README.md)）。

## 构建与测试

### 1. 工具链

- **x86_64**：无需额外配置，使用宿主机或 devcontainer 自带编译器。
- **aarch64-linux-gnu_orange_pi_4_pro_a733**：下载 `gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu`，放置于 `/opt/toolchains/`
  - 下载：https://mirrors.tuna.tsinghua.edu.cn/armbian-releases/_toolchain/
- **armv7l-linux-gnueabihf_comake_pi_d1_ssd2355**：`arm-unknown-linux-gcc-12.4.0-glibc-2.37-gnueabihf`，放置于 `/opt/toolchains/`（见 sigmastar devcontainer）

### 2. 第三方库

构建产物安装至 `third_party/<platform>/`（`<platform>` 与 preset 对应 toolchain 中的平台 tag 一致）。

- **公开库**（zlib、cnpy、gtest、opencv、spdlog 等）：由 `scripts/third_party_builder.sh` 调度，各库的下载与编译脚本在 [`scripts/third_party_builders/`](scripts/third_party_builders/)（`builder_*.sh`），可直接构建。
- **`--link static|shared`**：控制 gtest、opencv、spdlog、yaml-cpp 等为静态或动态库，**默认 static**（**zlib、cnpy 例外**：上游同时安装 `.a`/`.so`，不受 `--link` 影响）。
- **install 运行时库**：
  - **zlib / cnpy**：由 CMake 选项 `ZLIB_LINK_TYPE`、`CNPY_LINK_TYPE` 选择链哪种；设为 `SHARED` 时 `cmake --install` 安装对应 `.so` 到 `install/<platform>/lib/`。
  - **gtest / opencv / spdlog / yaml-cpp**：由 `find_package` 判定（`SHARED IMPORTED` 或 `OpenCV_SHARED`）；third_party 用 `--link shared` 构建后，`cmake --install` 自动安装所需 `.so`，无需额外 CMake 参数。
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
bash scripts/build.sh --preset <preset> --install
```

```bash
bash scripts/build.sh --preset x86_64-debug --install
bash scripts/build.sh --preset aarch64-linux-gnu_orange_pi_4_pro_a733-release --install
```

install 包布局（`install/<platform>/`）：

```
install/<platform>/
├── bin/                          # demo 可执行文件
├── lib/                          # 自研库、VIPLite .so；shared 第三方 .so（find_package 或 ZLIB/CNPY_LINK_TYPE=SHARED）
├── include/deploy_percept/       # 对外头文件
├── tests/                        # 测试可执行文件（INSTALL_TESTS）
├── benchmarks/                   # 延迟 benchmark（ENABLE/INSTALL_BENCHMARKS）
├── profiling/                    # 资源画像（ENABLE/INSTALL_PROFILING）
├── scripts/test_run.sh           # 板上/本机：对某目录逐个跑 ELF（sh）
└── share/percept/fixtures/       # 测试与 demo 共用数据
```

### 5. 测试

**PC 全流程**（build → install → **主机编排**跑 tests）：

```bash
bash scripts/test.sh --preset x86_64-debug
# 日志：ci-artifacts/host-logs/{orchestrator.log,tests/*.log,meta.txt}
```

**仅部署到开发板**（只 scp，不 build；默认 quick 清单，首次/CI 加 `--full`）：

```bash
export SSHPASS='your_password'
# 日常（默认非全量，清单见 scripts/sync_install_to_board.sh 内 QUICK_PATHS）
bash scripts/sync_install_to_board.sh \
  --install-dir install/armv7l-linux-gnueabihf_comake_pi_d1_ssd2355 \
  --board root@192.168.1.102:/customer/tmp/deploy_percept

# 首次上板或 CI：全量清空后整树上传
bash scripts/sync_install_to_board.sh \
  --install-dir install/armv7l-linux-gnueabihf_comake_pi_d1_ssd2355 \
  --board root@192.168.1.102:/customer/tmp/deploy_percept --full
```

约束：`--board` 路径须为**绝对路径**，且 basename 为 `deploy_percept`。

**开发板跑测**（build+install → 调 `sync_install_to_board.sh` → 主机逐条 ssh；板端无需 bash/rsync）：

```bash
export SSHPASS='your_password'   # 板端仅支持密码登录（需 sshpass）

# 日常（默认 quick 部署）
bash scripts/test.sh --preset armv7l-linux-gnueabihf_comake_pi_d1_ssd2355-release \
  --board root@192.168.1.102:/customer/tmp/deploy_percept

# 首次 / CI：加 --full
bash scripts/test.sh --preset armv7l-linux-gnueabihf_comake_pi_d1_ssd2355-release \
  --board root@192.168.1.102:/customer/tmp/deploy_percept --full

```

环境变量由主机拼进每条 remote 命令（`LD_LIBRARY_PATH` / `PERCEPT_ROOT` / `PERCEPT_OUTPUT_DIR`）。

**板上/本机跑某一目录全部测试**（`test_run.sh`，POSIX sh，无额外参数）：

```bash
sh scripts/test_run.sh /path/to/install/.../tests
# 板上（sync 后）:
sh /customer/tmp/deploy_percept/scripts/test_run.sh /customer/tmp/deploy_percept/tests
```

需要给某个 ELF 传参时，请直接单独运行该文件。

**板上手动跑单个测试**：

```bash
cd /customer/tmp/deploy_percept
export LD_LIBRARY_PATH=$PWD/lib PERCEPT_ROOT=$PWD/share/percept
./tests/test_yolov8_detect_sigmastar
```

NPU `core_loading` 在 debugfs 上，板端建议 **root SSH**（`--board root@host:/abs/deploy_percept`），见 [tests/profiling/README.md](tests/profiling/README.md)。`SSHPASS` 只用于 SSH。

板端跑完后日志在主机 `ci-artifacts/board-logs/`（`orchestrator.log`、`tests/*.log`、`meta.txt`；bench/profile 同分目录）。Board CI 上传 artifact `board-logs-<sha>`。

开发期 build tree 单测：`cd build/<preset> && ctest --output-on-failure`

## 架构说明

### 目录结构

```
deploy_percept/
├── CMakeLists.txt
├── CMakePresets.json              # configure / build preset
├── README.md
├── apps/                          # 示例应用
├── cmake/
│   ├── FindThirdParty.cmake       # 第三方库查找唯一入口
│   ├── modules/                   # Find*.cmake
│   └── toolchains/
├── include/deploy_percept/
│   ├── common/                    # types 等公共契约
│   ├── core/                      # engine、post_process、deploy_percept.hpp
│   ├── data/                      # COCO / ImageNet 标签
│   └── utils/                     # 预处理、可视化、io
├── scripts/
│   ├── build.sh                   # 构建（可选 --install）
│   ├── sync_install_to_board.sh   # 仅 scp install → 板（默认 quick，--full 全量）
│   ├── test.sh                    # 编排：build + install + 可选 sync + 调 test_run
│   ├── test_run.sh                # 对给定目录逐个跑 ELF（sh；本机=板端）
│   ├── third_party_builder.sh
│   └── third_party_builders/
├── src/
│   ├── common/                    # deploy_percept_common（INTERFACE）
│   ├── core/                      # deploy_percept_core
│   └── utils/                     # deploy_percept_utils
├── tests/
│   ├── fixtures/                  # install → share/percept/fixtures/
│   ├── test_common/               # 路径解析、fixture 工具（tests + benchmarks 共用）
│   ├── benchmarks/                # 性能 benchmark（AWNN，ENABLE_BENCHMARKS）
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
