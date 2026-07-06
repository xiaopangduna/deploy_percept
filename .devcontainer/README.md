# Dev Containers

每个子目录对应一种开发容器配置，无根目录默认入口；打开容器时需显式选择配置。

## 目录结构

```
.devcontainer/
├── README.md
├── ubuntu-22.04-allwinner/    # Allwinner / Orange Pi（A733 等）交叉编译
│   ├── devcontainer.json
│   ├── Dockerfile
│   ├── post-create.sh
│   ├── post-start.sh
│   └── .dockerignore
└── <future-container>/        # 后续新增容器
```

## 使用方式

1. 命令面板：`Dev Containers: Open Container Configuration File`
2. 选择 `.devcontainer/ubuntu-22.04-allwinner/devcontainer.json`
3. **Rebuild and Reopen in Container**（修改 devcontainer 配置后必须 Rebuild）

## ubuntu-22.04-allwinner

- 基础镜像：Ubuntu 22.04（orangepi-build 官方推荐宿主机版本）
- 运行用户：`vscode`（非 root；`updateRemoteUserUID` 与 WSL 宿主机 UID 对齐，避免 bind mount 出现 `nobody` 权限问题）
- 工作区：`deploy_percept` 作为独立仓库打开时，**不要**设置 `workspaceMount` / `workspaceFolder`（`devcontainer.json` 中默认已注释）
- 工具链：宿主机 `tmp/toolchains/` → 容器 `/opt/toolchains/`
- 预期工具链：`gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu`（A733 交叉编译）

### 工作区挂载

| 场景 | 做法 |
|------|------|
| **独立仓库**（默认） | 在 Cursor 打开 `deploy_percept` 根目录；`devcontainer.json` 里 `workspaceMount` / `workspaceFolder` **保持注释** |
| **嵌在 monorepo** | 取消注释上述两行，按实际上级路径修改 `source`、`target`、`workspaceFolder` |

工具链准备（解压或链接到项目内 `tmp/toolchains/`）：

```bash
mkdir -p tmp/toolchains
# 手动解压官方包，或链接已有工具链：
ln -sfn /path/to/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu \
  tmp/toolchains/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu
```

下载参考：https://mirrors.tuna.tsinghua.edu.cn/armbian-releases/_toolchain/
