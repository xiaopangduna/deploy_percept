# Dev Containers

每个子目录对应一种开发容器配置，无根目录默认入口；打开容器时需显式选择配置。平台细节、工具链准备与构建后验证见各子目录 README。

交叉工具链与容器解耦：容器只挂载 `/opt/toolchains` 并提供通用构建/联调工具；选哪条链由 `cmake/toolchains/*.cmake` 与 CMake preset / `third_party_builder` 决定。

## 目录结构

```
.devcontainer/
├── README.md
├── ubuntu-22.04-allwinner/    # Allwinner / Orange Pi（A733 等）交叉编译
│   └── README.md
└── ubuntu-20.04-sigmastar/    # Sigmastar 通用（CoMake D1 / SSC308QE 等）
    └── README.md
```

- [ubuntu-22.04-allwinner](ubuntu-22.04-allwinner/README.md)
- [ubuntu-20.04-sigmastar](ubuntu-20.04-sigmastar/README.md)

## 使用方式

1. 以 `deploy_percept` 为工作区根打开工程。
2. 命令面板：`Dev Containers: Open Container Configuration File`
3. 选择目标配置：
   - `.devcontainer/ubuntu-22.04-allwinner/devcontainer.json`
   - `.devcontainer/ubuntu-20.04-sigmastar/devcontainer.json`
4. **Rebuild and Reopen in Container**（修改 Dockerfile / `devcontainer.json` 后必须 Rebuild）

## 共用约定

- 运行用户：`vscode`（`common-utils` + `updateRemoteUserUID`，与 WSL 宿主机 UID 对齐，避免 bind mount 出现 `nobody`）
- 工作区：挂载 `deploy_percept` 上两级目录（`deep_learning`）到 `/workspace/deep_learning`；容器内工作目录为 `/workspace/deep_learning/src/deploy_percept`
- 工具链：宿主机 `/opt/toolchains/` → 容器 `/opt/toolchains/`；路径以各平台 `cmake/toolchains/*-toolchain.cmake` 为准
- 运行网络：`--network=host`；X11 透传（`DISPLAY`、`/tmp/.X11-unix`）
- SSH：挂载宿主机 `~/.ssh` 到 `/home/vscode/.ssh`

新增平台时：在 `cmake/toolchains/` 与 `CMakePresets.json` 增加条目即可；不必为每个芯片再建一个 Dev Container（除非 OS/基镜像不同，如 Allwinner 用 22.04）。
