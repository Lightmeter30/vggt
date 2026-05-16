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

## 数据与预处理

- Co3D annotation：`data/preprocess/generate_local_co3d_annotations.py`
- RealEstate10K 抽帧与 manifest：`data/preprocess/generate_local_realestate10k_frames.py`
- EuRoC Dataset：`data/datasets/euroc.py`
- Co3D Dataset：`data/datasets/co3d.py`

## 规则

- Hydra 路径和相对导入依赖从 `training/` 目录启动。
- 修改 Dataset 时保持 OpenCV `camera-from-world` 约定。
- 修改配置时保留用户需要替换的数据集路径占位项。
- 训练依赖本地数据集、checkpoint、GPU；不可用时说明限制。
- 缺依赖安装到 `my_vggt_relocation`。

## 验证

```bash
conda run -n my_vggt_relocation python -m py_compile training/launch.py
conda run -n my_vggt_relocation python -m py_compile training/trainer.py
conda run -n my_vggt_relocation python -m py_compile training/loss.py
conda run -n my_vggt_relocation python -m pytest test/ -q
```
