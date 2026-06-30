# yolov5_detect_awnn

全志 VIPLite YOLOv5 检测（Orange Pi 4 Pro A733）。推理经 `deploy_percept::engine::AwnnEngine`。

## 资源（install → share/percept/apps/yolov5_detect_awnn/）

| 文件 | 说明 |
|------|------|
| `yolov5.nb` | A733 模型 |
| `dog.jpg` | 默认输入图 |

## 构建与安装

```bash
cmake --preset=aarch64-linux-gnu_orange_pi_4_pro_a733-release \
  -DBUILD_AWNN_APPS=ON -DENABLE_BENCHMARKS=ON
cmake --build --preset=aarch64-linux-gnu_orange_pi_4_pro_a733-release \
  --target yolov5_detect_awnn bench_yolov5_full_pipeline bench_yolov5_preprocess
bash scripts/install.sh --preset aarch64-linux-gnu_orange_pi_4_pro_a733-release
```

## 运行

**install 树（板端）— demo：**

```bash
cd ~/deploy_percept
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
./bin/yolov5_detect_awnn
```

未设置 `PERCEPT_ROOT` 时，程序会从当前目录或可执行文件路径自动查找 `share/percept/apps/`。
也可显式设置：`export PERCEPT_ROOT=$PWD/share/percept`

**必须**保证 VIPLite 动态库来自 install 的 `lib/`（与构建版本一致）：

```bash
ls -la lib/libNBGlinker.so lib/libVIPhal.so
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH   # 建议始终设置
```

若出现 `Segmentation fault` 且只打印 VIPLite 版本号，常见原因：

1. `lib/` 未同步或混用了系统里其它版本的 `libVIPhal.so`
2. 可执行文件与 `libdeploy_percept.a` 不同步 — 请完整 rebuild + install + rsync

诊断日志（stderr）会逐步打印 `AwnnEngine: vip_create_network ok` 等；请记下**最后一条**再反馈。

默认输出：`share/percept/apps/yolov5_detect_awnn/yolov5_detect_awnn_out.jpg`

**性能 benchmark（全流程，install → `share/percept/benchmarks/`）：**

```bash
./share/percept/benchmarks/bench_yolov5_full_pipeline          # 输出路径对比
./share/percept/benchmarks/bench_yolov5_preprocess             # 预处理对比
./share/percept/benchmarks/bench_yolov5_full_pipeline 100      # 指定 loops

# 或 PC 远程：
bash scripts/bench.sh --board orangepi@<ip>:~/deploy_percept -- 50
```

正确性回归见 `share/percept/tests/test_yolov5_detect_awnn`（integration test）。

## 模型 I/O

- 输入：UINT8 NCHW 640×640
- 输出：`yolov5.nb` 为三头 **FP32**（stride 8/16/32），后处理用 `YoloV5DetectPostProcessAwnn`
