# AGENTS.md

## 范围

- `main.py`: 统一评测 CLI。
- `tasks.py`: dataset / task 注册。
- `common/`: 模型加载、设备 / dtype、指标、IO。
- `datasets/`: `co3d`、`euroc`、`realestate10k`、`realx3d` 评测实现。
- `README.md`: 评测命令与数据准备说明。

## 支持任务

- `co3d / camera_pose`
- `euroc / camera_pose`
- `realestate10k / camera_pose`
- `realx3d / camera_pose`

当前不支持：

- `image_matching`
- Bundle Adjustment 评测

## 通用命令

```bash
conda run -n my_vggt_relocation python evaluation/main.py \
    --dataset <dataset> \
    --task camera_pose \
    --model_path /home/zwr/code/my-vggt/ckpt/model.pt
```

```bash
conda run -n my_vggt_relocation python -m pytest test/test_evaluation.py -q
```

## RealEstate10K

```bash
conda run -n my_vggt_relocation python training/data/preprocess/generate_local_realestate10k_frames.py \
    --realestate10k_dir dataset/realEstate10K \
    --split test \
    --overwrite
```

```bash
conda run -n my_vggt_relocation python evaluation/main.py \
    --dataset realestate10k \
    --task camera_pose \
    --model_path /home/zwr/code/my-vggt/ckpt/model.pt \
    --realestate10k_dir dataset/realEstate10K \
    --frame_manifest_path dataset/realEstate10K/transcode_test_manifest.jsonl \
    --split test \
    --fast_eval \
    --num_frames 10 \
    --require_frame_manifest
```

- 使用仓库内 `generate_local_realestate10k_frames.py` 生成 manifest。
- 不要复用旧版 `downloadAndProcess.py` 或 `myDownloadAndProcess.py` 生成的 `transcode/`。

## RealX3D

```bash
CUDA_VISIBLE_DEVICES=n conda run -n my_vggt_relocation python evaluation/main.py \
    --dataset realx3d \
    --task camera_pose \
    --model_path /home/zwr/code/my-vggt/ckpt/model.pt \
    --data_root dataset/realX3D/data_4 \
    --conditions lowlight \
    --scenes MilkCookie \
    --splits train val \
    --max_frames 5 \
    --output_dir outputs/realx3d_smoke
```

- GT 来自 Blender / NeRF 风格 `camera-to-world`。
- 评测代码转换到 OpenCV c2w。
- VGGT 输出 OpenCV `camera-from-world` 后再转换到 c2w。
- 用 Umeyama Sim(3) 对齐到 GT 坐标系。
- 修改前必须确认左右坐标基变换和 pose 方向。

## Co3D / EuRoC

```bash
conda run -n my_vggt_relocation python training/data/preprocess/generate_local_co3d_annotations.py \
    --co3d_dir /YOUR/LOCAL/CO3D/PATH \
    --output_dir /home/zwr/code/my-vggt/dataset/co3d-anno-local
```

```bash
conda run -n my_vggt_relocation python evaluation/main.py \
    --dataset euroc \
    --task camera_pose \
    --model_path /home/zwr/code/my-vggt/ckpt/model.pt \
    --euroc_dir /YOUR/EUROC/PATH \
    --euroc_anno_dir /home/zwr/code/my-vggt/dataset/euroc-anno-local
```

## 规则

- 新 dataset / task 必须同步 `TASK_REGISTRY`。
- checkpoint 加载需兼容训练 checkpoint 嵌套键和 `module.` 前缀。
- 明确指标是相对位姿误差还是绝对位姿误差。
- 输出路径不得无提示覆盖用户已有结果。
- 完整评测依赖本地数据集、checkpoint、GPU；不可用时说明限制。
