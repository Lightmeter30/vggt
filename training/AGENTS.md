# AGENTS.md

## 范围

- `launch.py`: 训练入口，默认在 `training/` 目录内执行。
- `trainer.py`: 训练主循环。
- `loss.py`: 训练损失。
- `config/`: Hydra 配置。
- `data/`: Dataset、DataLoader、预处理。
- `train_utils/`: checkpoint、optimizer、logging、DDP 工具。
- `LOCAL_FINETUNE.md`, `README.md`: 训练说明。

## 入口命令

```bash
cd training && conda run -n my_vggt_relocation torchrun --nproc_per_node=4 launch.py
```

## 常查配置

- `config/default.yaml`
- `config/local_finetune.yaml`
- `config/my_finetune.yaml`
- `config/euroc_camera_only.yaml`: EuRoC camera-only 微调，camera loss 不依赖 `point_masks` 有效点门控。
- `config/euroc_degraded_baseline.yaml`: 在 camera-only 基线基础上启用 EuRoC 在线/固定视觉退化。
- `config/euroc_imu_film.yaml`: EuRoC IMU + FiLM 融合训练配置；默认保守 smoke 设置为 `img_size: 224`、`max_img_per_gpu: 2`、`img_nums: [2, 2]`。
- `config/degraded_finetune.yaml`
- `CO3D_DIR`
- `CO3D_ANNOTATION_DIR`
- `EUROC_DIR`
- `EUROC_ANNOTATION_DIR`
- `checkpoint.resume_checkpoint_path`

## 默认策略

- 当前默认配置偏微调。
- camera / depth 开启。
- point / track 关闭。
- `aggregator` 默认冻结。
- `my_finetune.yaml` 包含本机绝对路径；不要误改成临时不可复用路径。
- EuRoC VI 起步优先使用 camera-only；不要把 placeholder depth/point 当作真实监督。
- `euroc_imu_film.yaml` 会额外训练 `imu_encoder` 和 `imu_fusion`，gradient clip 配置必须覆盖这两个模块。

## 数据与预处理

- Co3D annotation：`data/preprocess/generate_local_co3d_annotations.py`
- RealEstate10K 抽帧与 manifest：`data/preprocess/generate_local_realestate10k_frames.py`
- EuRoC Dataset：`data/datasets/euroc.py`
- Co3D Dataset：`data/datasets/co3d.py`
- EuRoC IMU 字段来自 `EurocDataset(load_imu=True)`，训练 batch 里使用 `imu_windows`、`imu_window_masks`。
- 视觉退化优先看 `data/degradation.py`、`data/preprocess/generate_degraded_images.py`、`data/preprocess/visualize_degradation_samples.py`。

## 规则

- Hydra 路径和相对导入依赖从 `training/` 目录启动。
- 修改 Dataset 时保持 OpenCV `camera-from-world` 约定。
- 修改配置时保留用户需要替换的数据集路径占位项。
- 训练依赖本地数据集、checkpoint、GPU；不可用时说明限制。
- 缺依赖安装到 `my_vggt_relocation`。
- `trainer.py` 调用模型时会透传可选 `imu_windows`、`imu_window_masks`、`degradation_metadata`；改 batch 字段名时必须同步这里和测试。
- `loss.compute_camera_loss()` 的 `use_point_mask=False` 是 EuRoC camera-only 的关键配置，改动前先看 `test/test_camera_loss.py`。
- 4 张 4090 上 518 分辨率的训练显存边界见 `my_docs/gpu_usage.md`；该目录默认被 Git 忽略，文档路径仅作本地参考。

## 验证

```bash
conda run -n my_vggt_relocation python -m py_compile training/launch.py
conda run -n my_vggt_relocation python -m py_compile training/trainer.py
conda run -n my_vggt_relocation python -m py_compile training/loss.py
conda run -n my_vggt_relocation python -m pytest test/test_camera_loss.py test/test_training_imu_integration.py test/test_training_configs.py -q
conda run -n my_vggt_relocation python -m pytest test/ -q
```
