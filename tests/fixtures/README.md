# tests/fixtures

测试与 demo 共用的静态数据。源码树位于 `tests/fixtures/`；`cmake --install` 会安装到 `share/percept/fixtures/`。

## 目录

```text
tests/fixtures/
├── unit/post_process/          # 后处理单测 golden（无需 NPU）
│   ├── yolov5_detect/          model_outputs.npz, bus.jpg
│   ├── yolov5_seg/             model_outputs.npz, mask.bin, bus.jpg
│   └── yolov8_seg/             model_outputs.npz, mask.bin, bus.jpg
└── integration/awnn/           # AWNN 集成测 / demo / benchmark
    ├── yolov5_detect/          yolov5.nb, dog.jpg
    ├── yolov8_detect/          yolov8.nb, dog.jpg
    └── mobilenetv2_classify/   mobilenetv2.nb, cat.jpg
```

`unit/post_process/*/model_outputs.npz` 由 RKNN 导出得到，单测只比对后处理逻辑，运行时不依赖 RKNN。

## 路径 API

```cpp
#include "deploy_percept/utils/fixture_path.hpp"

deploy_percept::utils::fixture_path("integration/awnn/yolov5_detect/yolov5.nb");
```

测试内可用 `percept::test::fixture_data("unit/post_process/yolov5_detect/model_outputs.npz")`。

## 解析规则

| 场景 | `PERCEPT_ROOT` | 实际读取 |
|------|----------------|----------|
| `ctest`（build tree） | 项目根 | `tests/fixtures/...` |
| install / `test.sh` | `share/percept` | `share/percept/fixtures/...` |

未设置 `PERCEPT_ROOT` 时，会从可执行文件路径向上查找 `share/percept` 或项目根。
