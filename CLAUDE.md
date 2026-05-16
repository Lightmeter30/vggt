# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 环境

所有 Python 命令必须通过 conda 环境 `my_vggt_relocation` 运行：

```bash
conda run -n my_vggt_relocation python <script>
conda run -n my_vggt_relocation python -m py_compile <file>
conda run -n my_vggt_relocation pip install <package>
```

禁止直接使用 `python`、`pip` 或 `pytest`。

## 常用命令

### 语法检查（最小验证）

```bash
conda run -n my_vggt_relocation python -m py_compile vggt/models/vggt.py
```

### 运行测试

```bash
# 全部测试
conda run -n my_vggt_relocation python -m pytest test/ -q

# 单个测试文件
conda run -n my_vggt_relocation python -m pytest test/test_evaluation.py -v

# 单个测试类
conda run -n my_vggt_relocation python -m pytest test/test_evaluation.py::TestEvaluationMetrics -v
```

### 评测（相机位姿）

EuRoC：

```bash
conda run -n my_vggt_relocation python evaluation/main.py \
    --dataset euroc \
    --task camera_pose \
    --model_path /home/zwr/code/my-vggt/ckpt/model.pt \
    --euroc_dir /YOUR/EUROC/PATH \
    --euroc_anno_dir /home/zwr/code/my-vggt/dataset/euroc-anno-local
```

RealEstate10K（先抽帧，再评测）：

```bash
conda run -n my_vggt_relocation python training/data/preprocess/generate_local_realestate10k_frames.py \
    --realestate10k_dir dataset/realEstate10K --split test --overwrite

conda run -n my_vggt_relocation python evaluation/main.py \
    --dataset realestate10k --task camera_pose \
    --model_path /home/zwr/code/my-vggt/ckpt/model.pt \
    --realestate10k_dir dataset/realEstate10K \
    --split test --fast_eval --num_frames 10 --require_frame_manifest
```

RealX3D（完整评测）：

```bash
CUDA_VISIBLE_DEVICES=n conda run -n my_vggt_relocation python evaluation/main.py \
    --dataset realx3d \
    --task camera_pose \
    --model_path /home/zwr/code/my-vggt/ckpt/model.pt \
    --data_root dataset/realX3D/data_4 \
    --max_frames 10 \
    --output_dir outputs/realx3d_all_results
```

### Demo

```bash
# Gradio Web Demo（需要 CUDA 和联网下载权重）
conda run -n my_vggt_relocation python demo_gradio.py

# Viser 3D 可视化
conda run -n my_vggt_relocation python demo_viser.py --image_folder examples/kitchen/images

# 导出 COLMAP
conda run -n my_vggt_relocation python demo_colmap.py --scene_dir examples/kitchen
```

### 训练（需在 training/ 目录内执行）

```bash
cd training && conda run -n my_vggt_relocation torchrun --nproc_per_node=4 launch.py
```

## 架构概览

VGGT (Visual Geometry Grounded Transformer, CVPR 2025 Best Paper) 是一个前馈神经网络，
从一张、几张或数百张视图直接推断场景的相机内外参、点图、深度图和 3D 轨迹。

核心推理流程：

```
输入: [S, 3, H, W] 或 [B, S, 3, H, W] 图像
       ↓
model.aggregator(images)  → aggregated_tokens_list, ps_idx
       ↓
model.camera_head(...)    → pose_enc → extrinsic [3×4], intrinsic [3×3]
model.depth_head(...)     → depth_map, depth_conf
model.point_head(...)     → point_map, point_conf
model.track_head(...)     → track, vis_score, conf_score
```

### 核心模型 (`vggt/`)

| 模块 | 说明 |
|------|------|
| `vggt/models/vggt.py` | 顶层 VGGT 类，组合 aggregator + 四个任务头，加载 HuggingFace 权重 |
| `vggt/models/aggregator.py` | 共享特征提取 Transformer，所有任务头的输入来源 |
| `vggt/heads/camera_head.py` | 相机头 — 输出 pose_enc |
| `vggt/heads/dpt_head.py` | 深度头 — 输出 depth_map, depth_conf |
| `vggt/heads/track_head.py` | 轨迹头 — 输出 track, vis_score, conf_score |
| `vggt/heads/head_act.py` | 点头激活 |
| `vggt/heads/utils.py` | 头模块工具函数 |
| `vggt/layers/` | Transformer、RoPE、patch embedding、SwiGLU FFN 等底层模块 |
| `vggt/utils/load_fn.py` | 推理图像预处理统一入口 |
| `vggt/utils/pose_enc.py` | 位姿编码与解码 |
| `vggt/utils/geometry.py` | 几何工具 |
| `vggt/utils/rotation.py` | 旋转矩阵与四元数转换 |
| `vggt/dependency/` | COLMAP 导出、点追踪、畸变处理、VGG-SfM tracker |

### 评测系统 (`evaluation/`)

```
evaluation/main.py                  → 统一 CLI 入口
evaluation/tasks.py                 → TASK_REGISTRY 调度
evaluation/common/model.py          → 模型加载、设备/dtype 解析
evaluation/common/metrics.py        → AUC、相对位姿误差等指标
evaluation/common/io.py             → 结果读写
evaluation/datasets/
  co3d/camera_pose.py               → Co3D 评测
  euroc/camera_pose.py              → EuRoC 评测
  realestate10k/camera_pose.py      → RealEstate10K 评测（含多 GPU 并行）
  realx3d/camera_pose.py            → RealX3D 评测（Sim(3) 对齐）
```

当前支持的数据集：`co3d`、`euroc`、`realestate10k`、`realx3d`

### 训练系统 (`training/`)

```
training/launch.py                  → 训练入口（须在 training/ 内执行）
training/trainer.py                 → 主训练循环
training/loss.py                    → 损失函数
training/config/default.yaml        → Hydra 默认配置
training/config/local_finetune.yaml → 本地微调配置
training/config/my_finetune.yaml    → 本地 EuRoC 微调配置
training/data/datasets/             → Co3D、EuRoC、VKitti Dataset 类
training/data/preprocess/           → 数据集预处理脚本
training/train_utils/               → checkpoint、optimizer、DDP、logging、freeze 等
```

### 入口脚本

| 文件 | 功能 |
|------|------|
| `demo_gradio.py` | Gradio Web Demo |
| `demo_viser.py` | Viser 3D 可视化 |
| `demo_colmap.py` | COLMAP 格式导出（可选 BA） |
| `visual_util.py` | GLB 构建、天空分割下载 |

## 关键约束

### 坐标系

全项目使用 OpenCV `camera-from-world` 约定（外参为 3×4 矩阵）。相机外参处理、深度反投影、
COLMAP 导出、训练 Dataset 接入、评测指标计算均须保持一致，不得混用 OpenGL / PyTorch3D 约定。

**RealX3D 特殊路径**：GT 来自 Blender/NeRF 风格 `camera-to-world`，评测代码会：
1. 将 Blender c2w 转换到 OpenCV c2w
2. VGGT 输出 OpenCV `camera-from-world` 后转换到 c2w
3. 用 Umeyama Sim(3) 对齐到 GT 坐标系
4. 计算绝对位姿误差（max(rot, trans) 或 rotation_only）

### 输入形状

`VGGT.forward()` 同时支持 `[S, 3, H, W]` 和 `[B, S, 3, H, W]`，不得破坏此约定。

### 图像预处理

- `load_and_preprocess_images()`: 从文件路径加载，常规推理
- `load_and_preprocess_images_from_objects()`: 从 PIL Image 或 numpy array 加载，评测常用
- `load_and_preprocess_images_square()`: COLMAP 导出路径常用
- 关键尺寸需能被 14 整除
- 注意 `crop` / `pad` 模式行为和 alpha 通道转白底

### RealEstate10K 抽帧

必须用仓库内的新抽帧脚本（`generate_local_realestate10k_frames.py`），旧脚本时间单位 bug 会导致
AUC 明显偏低。新脚本生成 `transcode/` 和 `transcode_test_manifest.jsonl`，评测时必须用
`--require_frame_manifest`。

### 训练配置

`training/config/default.yaml` 中的 `CO3D_DIR`、`CO3D_ANNOTATION_DIR`、`EUROC_DIR`、
`EUROC_ANNOTATION_DIR`、`resume_checkpoint_path` 是绝对路径占位项，修改时保留这些占位项。

### 模型权重下载

Demo 脚本首次运行会自动从 HuggingFace 下载 `facebook/VGGT-1B` 或 `facebook/VGGT-1B-Commercial`
权重；天空分割功能会下载 `skyseg.onnx`。修改 Demo 脚本前确认是否需要联网和 CUDA。

### 多 GPU 评测

RealEstate10K 支持 `--gpu_ids` 参数进行多 GPU 并行评测，通过 `multiprocessing.spawn`
每 GPU 独立加载模型并以 sequence 为单位分片处理。多 GPU 模式下主进程不加载模型（通过
`should_load_model()` 控制）。

## 信息优先级

文档与代码不一致时：源码 > README.md > evaluation/README.md > training/README.md > docs/package.md > my_docs/（本地阅读笔记）。

## 注意事项

- `examples/`、`input_images_*`、`.gradio/`、`.gradio_tmp/`、`my_docs/` 视为只读或本地产物，除非用户明确要求否则不要删除
- `outputs/`、`dataset/`、`ckpt/` 是本地产物，体积大且不易重建，不要随意清理
- `repomix-output.xml` 是只读快照，根据它理解仓库，但修改应落到真实源文件
- 修改 aggregator 或任务头后，同步检查 `forward()` 输出键名、demo 和评测中对 `predictions` 的消费、训练 loss 侧兼容性
- 如验证受限于网络、权重下载、GPU 或数据集，须在结果中明确说明
- 任何涉及相机、坐标系、点云导出、评测指标的修改，都要先确认数学约定
