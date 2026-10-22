# ubuntu-20.04-sigmastar

Sigmastar **通用**交叉编译 Dev Container（CoMake D1、SSC308QE 等共用）。共用约定见 [../README.md](../README.md)。

交叉工具链与容器解耦：容器只提供 OS / CMake / 联调工具，并挂载 `/opt/toolchains`；**选哪条链由** `cmake/toolchains/*.cmake` **与 CMake preset / `third_party_builder` 决定**，不在 Dockerfile 里写死 `TOOLCHAIN_ROOT`。

## 环境

- 基础镜像：`comake_d1_docker:latest`（需事先 `docker load`；Dockerfile 在其上叠一层）
- 下载：[CoMake SDK 支持页](https://www.comake.online/support/sdk/)（Docker 包 + SDK）
- CMake：Kitware 源（≥3.21，支持 `cmake --preset`）
- 工具链目录：宿主机 `/opt/toolchains/` → 容器 `/opt/toolchains/`

## 前置

### 1. 导入基镜像

```bash
docker images | grep comake_d1_docker
# 若无：docker load -i <path>/comake_d1_docker.tar.gz
```

`comake_d1_docker.tar.gz` 是 `docker save` 包，不要当普通压缩包解压。

### 2. 按需解压工具链到 `/opt/toolchains/`

压缩包**原名**直接解压到 `/opt/toolchains/<包名>/`，路径须与对应 cmake toolchain 一致：

| 平台 | cmake toolchain | `/opt/toolchains/` 下目录 |
| --- | --- | --- |
| CoMake Pi D1 (SSD2355) | `armv7l-linux-gnueabihf_comake_pi_d1_ssd2355-toolchain.cmake` | `arm-unknown-linux-gcc-12.4.0-glibc-2.37-gnueabihf` |
| SSC308QE (Iford) | `armv7l-linux-gnueabihf_ssc308qe-toolchain.cmake` | `gcc-11.1.0-20210608-sigmastar-glibc-x86_64_arm-linux-gnueabihf` |

示例：

```bash
sudo mkdir -p /opt/toolchains
# CoMake D1
tar -C /opt/toolchains -xf arm-unknown-linux-gcc-12.4.0-glibc-2.37-gnueabihf.tar.xz
# SSC308QE（Iford_IMSSV05C13/Tools/ToolChain）
tar -C /opt/toolchains -xf gcc-11.1.0-20210608-sigmastar-glibc-x86_64_arm-linux-gnueabihf.tar.xz
```

可同时存在多套；构建时用 preset / `--platform` 选择。

## 构建后验证

```bash
pwd
# 期望：/workspace/deep_learning/src/deploy_percept

cmake --version
# 期望：≥ 3.21

ls /opt/toolchains

# 按平台选 preset（绝对路径编译器，无需容器 PATH）
cmake --preset armv7l-linux-gnueabihf_comake_pi_d1_ssd2355-release
# 或
cmake --preset armv7l-linux-gnueabihf_ssc308qe-release
```

## 构建注意

- apt 源：Dockerfile 将 `archive` / `security` 换为清华镜像。
- 构建网络：`build.options: ["--network=host"]`（Clash TUN/fake-ip 时默认 bridge 易失败）。
- 加装工具：编辑本目录 `Dockerfile` 后 Rebuild。
