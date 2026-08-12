# tests/fixtures

测试与 demo 共用的静态数据。源码树位于 `tests/fixtures/`；`cmake --install` 会安装到 `share/percept/fixtures/`。

## 目录

```text
tests/fixtures/
├── unit/post_process/          # 后处理单测 golden（无需 NPU）
│   ├── yolov5_detect/          model_outputs.npz, bus.jpg（RKNN INT8）
│   ├── yolov5_detect_awnn/     model_outputs.npz（AWNN FP32，板端 dump）
│   ├── yolov5_seg/             model_outputs.npz, mask.bin, bus.jpg
│   ├── yolov8_detect_awnn/     model_outputs.npz（AWNN FP32，板端 dump）
│   ├── yolov8_seg/             model_outputs.npz, mask.bin, bus.jpg
│   └── mobilenetv2_classify_awnn/  model_outputs.npz（AWNN FP32，板端 dump）
├── integration/awnn/           # AWNN 集成测 / demo / benchmark
│   ├── yolov5_a733.nb
│   ├── yolov8_a733.nb
│   ├── mobilenetv2_a733.nb
│   ├── dog.jpg
│   └── cat.jpg
└── integration/sigmastar/      # Sigmastar IPU 集成测 / demo / bench_ipu_img
    ├── yolov8n_640x640_ssd2355.img
    ├── yolov8n_seg_640x640_ssd2355.img
    ├── mobilenet_v2_224_ssd2355.img
    ├── dog.jpg
    ├── bus.jpg
    └── apple.jpg
```

`unit/post_process/yolov5_detect/model_outputs.npz` 由 RKNN 导出（INT8），单测只比对后处理逻辑，运行时不依赖 RKNN。

`unit/post_process/yolov5_detect_awnn/model_outputs.npz` 与 `yolov8_detect_awnn/` 下同名文件由板端 AWNN 推理导出（FP32），输入须为 `integration/awnn/dog.jpg`，与 integration golden 对齐。**若 fixture 由旧版 `save_npz` 导出（误按 int8 写入），需用修复后的 demo 重新 dump。**

```bash
yolov5_detect_awnn yolov5_a733.nb dog.jpg out.jpg model_outputs.npz
yolov8_detect_awnn yolov8_a733.nb dog.jpg out.jpg model_outputs.npz
mobilenetv2_classify_awnn mobilenetv2_a733.nb cat.jpg model_outputs.npz
```

`unit/post_process/mobilenetv2_classify_awnn/model_outputs.npz` 输入须为 `integration/awnn/cat.jpg`，单路 `output_0`（1000×FP32 logits），golden `class_id=281` 与 integration 一致。

`integration/sigmastar/`：CoMake Pi D1 / pcupid 开放模型 `yolov8n_640x640_ssd2355.img`（检测）+ `dog.jpg`、`yolov8n_seg_640x640_ssd2355.img`（分割）+ `bus.jpg`、`mobilenet_v2_224_ssd2355.img`（分类，ImageNet-1000）+ `apple.jpg`。集成测、`benchmarks/bench_ipu_img`（无参自动扫本目录 `*.img`）与全流程 `bench_yolov8` / `bench_yolov8_seg` / `bench_mobilenetv2_classify` 需板端 IPU 固件 `/config/dla/ipu_firmware.bin`。

## 路径 API

```cpp
#include "test_common/paths.hpp"

percept::test::fixture_data("integration/awnn/yolov5_a733.nb");
```

测试内可用 `percept::test::fixture_data(...)`；benchmark 同样通过 `test_common/paths.hpp` 解析 fixtures。

## 解析规则

| 场景 | `PERCEPT_ROOT` | 实际读取 |
|------|----------------|----------|
| `ctest`（build tree） | 项目根 | `tests/fixtures/...` |
| install / `test.sh` | `share/percept` | `share/percept/fixtures/...` |

未设置 `PERCEPT_ROOT` 时，会从可执行文件路径向上查找 install prefix，再定位 `share/percept/fixtures/`（测试在 `tests/`，benchmark 在 install 的 `benchmarks/`）。
