# AGENTS.md

## 交流

- 始终使用中文与用户交流。
- 先看 `git status`，不要覆盖用户已有修改或本地产物。
- 默认小范围修改；不要顺手重构无关代码。

## 项目结构

- `vggt/`: 核心 Python 包；模型、heads、layers、几何工具。
- `evaluation/`: 相机位姿评测入口、任务注册、数据集实现。
- `training/`: Hydra 训练、微调、loss、数据加载；当前包含 EuRoC camera-only、退化 baseline、IMU FiLM 训练配置。
- `test/`: 单元测试，重点覆盖评测指标、数据加载、RealX3D 对齐，以及 EuRoC 训练配置、IMU 编码和视觉-惯性融合。
- `examples/`: 示例素材，默认只读。
- `docs/`, `my_docs/`: 文档与本地分析笔记；源码事实优先于笔记。
- `demo_gradio.py`: Gradio Web Demo。
- `demo_viser.py`: Viser 3D 浏览器。
- `demo_colmap.py`: COLMAP 导出。
- `visual_util.py`: GLB 构建、天空分割、可视化辅助。

## 信息优先级

1. 源码
2. `README.md`
3. `evaluation/README.md`
4. `training/README.md`
5. `docs/package.md`
6. `my_docs/`

## Python 环境

- 必须使用 Conda 环境：`my_vggt_relocation`。
- 不要直接运行系统 `python`、`pip`、`pytest`。
- Python 命令统一写成：`conda run -n my_vggt_relocation python ...`
- pip 命令统一写成：`conda run -n my_vggt_relocation pip ...`
- 修改 Python 代码前，先验证解释器：

```bash
conda run -n my_vggt_relocation python --version
```

## 常用命令

```bash
conda run -n my_vggt_relocation python -m py_compile <file.py>
conda run -n my_vggt_relocation python -m pytest test/ -q
conda run -n my_vggt_relocation python -m pytest test/test_evaluation.py -q
conda run -n my_vggt_relocation pip install -r requirements.txt
conda run -n my_vggt_relocation pip install -r requirements_demo.txt
```

## 推理与 Demo

```bash
conda run -n my_vggt_relocation python demo_gradio.py
conda run -n my_vggt_relocation python demo_viser.py --image_folder examples/kitchen/images
conda run -n my_vggt_relocation python demo_colmap.py --scene_dir examples/kitchen
conda run -n my_vggt_relocation python demo_colmap.py --scene_dir examples/kitchen --use_ba
```

- Demo 可能下载 `facebook/VGGT-1B`、`skyseg.onnx` 等模型。
- Demo 可能写入 `.gradio_tmp/`、`input_images_*`、`predictions.npz`、`.glb`、`scene_dir/sparse/`。
- Gradio 默认需要 CUDA；COLMAP BA 依赖 `pycolmap` 及追踪相关依赖。

## 核心约定

- `VGGT.forward()` 支持 `[S, 3, H, W]` 和 `[B, S, 3, H, W]`。
- `VGGT.forward()` 可选接收 `imu_windows`、`imu_window_masks`、`degradation_metadata`；只有 `model.imu.enabled=True` 时才要求 IMU 张量。
- IMU FiLM 融合在 `Aggregator` patch tokens 进入 attention 前执行；默认初始化应保持视觉 token 恒等映射。
- 训练、评测、导出默认使用 OpenCV `camera-from-world`。
- 不要无说明混入 OpenGL / PyTorch3D 坐标约定。
- 图像预处理统一优先看 `vggt/utils/load_fn.py`。
- 输入尺寸相关逻辑必须考虑 patch size 当前需能被 14 整除。
- alpha 通道按现有预处理约定转白底。
- EuRoC camera-only 训练可关闭 camera loss 的 `point_masks` 门控，避免 placeholder depth/point mask 让相机监督被整体跳过。

## 目录级规则

- 改核心模型或几何工具：先读 `vggt/AGENTS.md`。
- 改评测入口、指标、数据集：先读 `evaluation/AGENTS.md`。
- 改训练、微调、Hydra 配置：先读 `training/AGENTS.md`。

## 不要随意改动

- `examples/`
- `input_images_*`
- `.gradio/`, `.gradio_tmp/`
- `outputs/`
- `dataset/`
- `ckpt/`
- `repomix-output.xml`

除非用户明确要求，不要删除或清理以上内容。

## 修改规则

- `repomix-output.xml` 只读参考；修改真实源文件。
- 涉及相机、坐标系、点云导出、评测指标时，先确认数学约定。
- 修改训练配置时，保留用户需要替换的路径占位项。
- 修改 IMU 或视觉-惯性融合时，同步检查 `vggt/models/imu_encoder.py`、`vggt/models/visual_imu_fusion.py`、`vggt/models/vggt.py`、`vggt/models/aggregator.py` 和相关测试。
- 缺依赖时安装到 `my_vggt_relocation`，不要全局安装。
- 验证受限于网络、权重、GPU、数据集或磁盘输出时，在结果中说明。

## 最小排查顺序

1. `git status`
2. 读相关 README 或目录级 `AGENTS.md`
3. 读入口脚本、核心模块、配置
4. 做最小可行修改
5. 运行 `py_compile`、相关 `pytest` 或 smoke test
6. 汇报修改、验证、剩余风险
