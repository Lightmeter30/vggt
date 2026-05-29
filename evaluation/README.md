# Evaluation

统一评测入口位于 [main.py](/home/zwr/code/my-vggt/evaluation/main.py)。

当前支持：

- `co3d / camera_pose`
- `euroc / camera_pose`
- `realestate10k / camera_pose`
- `realx3d / camera_pose`

当前不支持：

- `image_matching`
- Bundle Adjustment 评测

## 运行方式

必须使用仓库指定环境：

运行前可以指定GPU：
```bash
CUDA_VISIBLE_DEVICES=n conda run ...
```

### Attention 可视化

```bash
conda run -n my_vggt_relocation python evaluation/visualize_attention.py \
    --image_dir /PATH/TO/images \
    --model_path /YOUR/MODEL/PATH \
    --output_dir outputs/attention_vis

conda run -n my_vggt_relocation python evaluation/compose_attention_grid.py \
    --run_dir outputs/attention_vis/run_000
```

`visualize_attention.py` 会写出 `manifest.json` 和逐层 attention overlay；`compose_attention_grid.py` 可把同一 run 合成网格图。

### EuRoC

```bash
conda run -n my_vggt_relocation python evaluation/main.py \
    --dataset euroc \
    --task camera_pose \
    --model_path /YOUR/MODEL/PATH \
    --euroc_dir /YOUR/EUROC/PATH \
    --euroc_anno_dir /home/zwr/code/my-vggt/dataset/euroc-anno-local
```


EuRoC 可选参数：

- `--split test`
- `--camera_names cam0`
- `--num_frames 10`
- `--min_num_images 24`
- `--fast_eval`
- `--no-undistort`
- `--use_imu`：从 EuRoC annotation 构造 IMU window，供 IMU-FiLM checkpoint 使用。
- `--metrics_output_dir evaluation/results`：写出 EuRoC 指标报告。

### RealEstate10K

RealEstate10K 相机位姿评测前，先用仓库内的抽帧脚本生成 `transcode/` 和
`transcode_manifest.jsonl`：

```bash
conda run -n my_vggt_relocation python training/data/preprocess/generate_local_realestate10k_frames.py \
    --realestate10k_dir dataset/realEstate10K \
    --split test \
    --overwrite
```

注意：不要直接复用旧版 `dataset/realEstate10K/downloadAndProcess.py` 或
`myDownloadAndProcess.py` 生成的 `transcode/`。旧脚本曾使用 `1e9 / (2 * fps)` 作为匹配窗口，
而 RealEstate10K 的 txt 时间戳单位是微秒；这会把大量 pose 时间戳错误匹配到同一个视频帧，
导致相机位姿评测的 RTA / AUC 明显偏低。新的抽帧脚本使用 `1e6 / (2 * fps)`，并通过
manifest 记录每张图的实际视频时间误差。

推荐评测命令：

```bash
conda run -n my_vggt_relocation python evaluation/main.py \
    --dataset realestate10k \
    --task camera_pose \
    --model_path /home/zwr/code/my-vggt/ckpt/model.pt \
    --realestate10k_dir dataset/realEstate10K \
    --frame_manifest_path dataset/realEstate10K/transcode_test_manifest.jsonl \
    --sequence_list_path dataset/realEstate10K/re10k_test_1800.txt \
    --split test \
    --num_frames 10 \
    --require_frame_manifest
```

RealEstate10K 额外可选参数：

- `--max_sequences N`
- `--thresholds 3 5 15 30`
- `--preprocess_mode crop|pad`：图像预处理方式，默认 `crop`；`crop` 将宽度缩放到 518 并在高度过大时中心裁剪，
  `pad` 保留完整画面并将短边补白到 518x518。
- `--frame_manifest_path /path/to/transcode_manifest.jsonl`
- `--require_frame_manifest`
- `--sequence_list_path dataset/realEstate10K/re10k_test_1800.txt`
- `--gpu_ids 4 5 6 7`：使用多个 CUDA GPU 以 sequence 为单位并行评测；不传时保持单 GPU/CPU 路径。
- `--metrics_output_path /path/to/report.txt`
- `--metrics_output_dir evaluation/results`

若要对齐 VGGT / PoseDiffusion 论文中的 RealEstate10K 1.8K 子集协议，
请传入 PoseDiffusion 的 `re10k_test_1800.txt`。该文件每行是一个 test metadata
文件名 stem，评测脚本会在 `test/<stem>.txt` 中过滤可用 sequence，并在报告中写出
`sequence_list_requested`、`sequence_list_matched`、`sequence_list_missing` 等统计。
不传该参数时，脚本仍使用本地可用的全量 test sequence。

### Co3D

Co3D 相机位姿评测：

```bash
conda run -n my_vggt_relocation python training/data/preprocess/generate_local_co3d_annotations.py \
    --co3d_dir /YOUR/LOCAL/CO3D/PATH \
    --output_dir /home/zwr/code/my-vggt/dataset/co3d-anno-local

conda run -n my_vggt_relocation python evaluation/main.py \
    --dataset co3d \
    --task camera_pose \
    --model_path /home/zwr/code/my-vggt/ckpt/model.pt \
    --co3d_dir /home/zwr/code/my-vggt/dataset/co3d \
    --co3d_anno_dir /home/zwr/code/my-vggt/dataset/co3d-anno-local \
    --split test \
    --categories all \
    --num_frames 10 \
    --min_num_images 10
```

Co3D 额外可选参数：

- `--categories all` 或 `--categories apple chair`
- `--max_sequences N`
- `--thresholds 3 5 15 30`
- `--preprocess_mode crop`
- `--metrics_output_path /path/to/report.txt`
- `--metrics_output_dir evaluation/results`

### RealX3D

RealX3D 相机位姿评测：

```bash
# smoke test（快速验证流程）
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

# 完整评测
CUDA_VISIBLE_DEVICES=n conda run -n my_vggt_relocation python evaluation/main.py \
    --dataset realx3d \
    --task camera_pose \
    --model_path /home/zwr/code/my-vggt/ckpt/model.pt \
    --max_frames 10 \
    --data_root dataset/realX3D/data_4 \
    --output_dir outputs/realx3d_all_results
```

RealX3D 可选参数：

- `--data_root` — RealX3D/data_4 根目录（必填）
- `--conditions lowlight smoke ...` — 要评测的退化条件，默认全部 9 种
- `--scenes Akikaze MilkCookie ...` — 要评测的场景，默认全部 15 个
- `--splits train val test` — 数据划分，默认 `train val`
- `--max_frames N` — 每序列最多使用帧数，调试用
- `--output_dir outputs/realx3d_vggt_pose` — 结果输出目录
- `--pose_error_mode max_rot_trans` — 位姿误差模式：`max_rot_trans`（默认）或 `rotation_only`
- `--resize_long_side N` — 预处理时限制图像长边最大尺寸

RealX3D 每次评测输出目录结构：

```
outputs/realx3d_vggt_pose/
├── summary.csv                    ← 全量明细（condition/scene/split 粒度）
├── summary_by_condition.csv       ← 按退化条件汇总
├── summary_by_split.csv           ← 按数据划分汇总
└── {condition}/{scene}/{split}/
    ├── per_frame_errors.csv       ← 逐帧误差
    ├── metrics.json               ← 序列级指标
    ├── pred_poses_aligned.npy     ← 对齐后的预测位姿 (N,4,4)
    ├── gt_poses.npy               ← GT 位姿 (N,4,4)
    └── config.json                ← 对齐参数 (s, R, t)
```

## 指标


### EuRoC

EuRoC 相机位姿评测沿用 VGGT 在 Co3D 上的相对位姿评测定义：

- 对每个 sequence 采样若干帧
- 计算所有帧对之间的相对旋转误差
- 计算所有帧对之间的相对平移方向误差
- 汇总 `AUC@30`、`AUC@15`、`AUC@5`、`AUC@3`

GT 外参直接使用 EuRoC annotation 中的 OpenCV `camera-from-world` `3x4` 外参。

RealEstate10K 外参直接使用官方 txt 中的 `3x4` camera pose。评测会从首行 YouTube URL 解析
`video_id`，并读取 `transcode/<video_id>/<timestamp>.jpg`。如果视频已过期或本地没有抽帧结果，
对应 txt 会被跳过；只有可用帧数不少于 `--min_num_images` 的 sequence 会进入评测。
如果 `transcode_manifest.jsonl` 存在，评测会自动校验每张图的抽帧时间误差；如果显式传入
`--require_frame_manifest`，缺少 manifest 或时间误差超过半帧窗口的图片都会被过滤，避免错误图像
和 pose 配对静默进入指标。

### Co3D

Co3D 外参直接使用
[training/data/preprocess/generate_local_co3d_annotations.py](/home/zwr/code/my-vggt/training/data/preprocess/generate_local_co3d_annotations.py)
生成的 annotation 中的 OpenCV `camera-from-world` `extri`。评测入口不会再执行
PyTorch3D 到 OpenCV 的二次转换。`--categories all` 会扫描 `--co3d_anno_dir`
下的 `*_test.jgz` 或 `*_train.jgz`，因此本地只有单个 category / 单个 sequence 的子集也可以运行；
空 annotation 文件会被自然跳过，只有本地图片存在且可用帧数满足 `--min_num_images` 和
`--num_frames` 的 sequence 会进入评测。

RealEstate10K 每次评测都会写出一份文本报告。默认路径为
`evaluation/results/realestate10k_camera_pose_<timestamp>.txt`，也可以通过
`--metrics_output_path` 指定完整输出文件路径。报告包含运行参数、summary 指标、每个 sequence
的帧索引、旋转/平移误差统计以及 `RRA@N`、`RTA@N`、`AUC@N`。

Co3D 每次评测同样会写出文本报告。默认路径为
`evaluation/results/co3d_camera_pose_<timestamp>.txt`。

### RealX3D

RealX3D 外参与其他数据集不同，使用绝对位姿误差评测：

1. 从 `transforms_{split}.json` 读取 Blender/NeRF 风格 GT `camera-to-world` (c2w) 位姿，并转换到 OpenCV c2w
2. VGGT 推理得到 OpenCV `camera-from-world` 外参，转换为 c2w
3. 对预测 c2w 的相机中心做 Umeyama Sim(3) 对齐到 GT 坐标系（求 scale s、rotation R、translation t）
4. 将对齐后的预测与 GT 逐帧计算旋转误差和平移方向角误差
5. 位姿误差 = max(旋转误差, 平移误差)，汇总 AUC@5/10/20

默认 `--splits train val` 会把 `train` 作为对应退化 condition 评测，同时为每个 condition
额外构造 `{condition}_clean` 任务：图像来自同场景 `val` split，帧名与本次选中的 train 帧一一对应。
这些 clean 任务的单序列输出目录仍写到 `{condition}_clean/{scene}/val/`，summary 中统一汇总为
`condition=clean`。

若某个 condition/scene/split 组合不存在，评测会自动跳过并记录 warning。
