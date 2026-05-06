# Evaluation

统一评测入口位于 [main.py](/home/zwr/code/my-vggt/evaluation/main.py)。

当前支持：

- `co3d / camera_pose`
- `euroc / camera_pose`
- `realestate10k / camera_pose`

当前不支持：

- `image_matching`
- Bundle Adjustment 评测

## 运行方式

必须使用仓库指定环境：

```bash
conda run -n my_vggt_relocation python evaluation/main.py \
    --dataset euroc \
    --task camera_pose \
    --model_path /YOUR/MODEL/PATH \
    --euroc_dir /YOUR/EUROC/PATH \
    --euroc_anno_dir /home/zwr/code/my-vggt/dataset/euroc-anno-local
```

RealEstate10K 相机位姿评测：

```bash
conda run -n my_vggt_relocation python evaluation/main.py \
    --dataset realestate10k \
    --task camera_pose \
    --model_path /home/zwr/code/my-vggt/ckpt/model.pt \
    --realestate10k_dir dataset/realEstate10K \
    --split test \
    --fast_eval \
    --num_frames 10
```

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

可选参数：

- `--split test`
- `--camera_names cam0`
- `--num_frames 10`
- `--min_num_images 24`
- `--fast_eval`
- `--no-undistort`

RealEstate10K 额外可选参数：

- `--max_sequences N`
- `--thresholds 3 5 15 30`
- `--preprocess_mode crop`
- `--metrics_output_path /path/to/report.txt`
- `--metrics_output_dir evaluation/results`

Co3D 额外可选参数：

- `--categories all` 或 `--categories apple chair`
- `--max_sequences N`
- `--thresholds 3 5 15 30`
- `--preprocess_mode crop`
- `--metrics_output_path /path/to/report.txt`
- `--metrics_output_dir evaluation/results`

## 指标

EuRoC 相机位姿评测沿用 VGGT 在 Co3D 上的相对位姿评测定义：

- 对每个 sequence 采样若干帧
- 计算所有帧对之间的相对旋转误差
- 计算所有帧对之间的相对平移方向误差
- 汇总 `AUC@30`、`AUC@15`、`AUC@5`、`AUC@3`

GT 外参直接使用 EuRoC annotation 中的 OpenCV `camera-from-world` `3x4` 外参。

RealEstate10K 外参直接使用官方 txt 中的 `3x4` camera pose。评测会从首行 YouTube URL 解析
`video_id`，并读取 `transcode/<video_id>/<timestamp>.jpg`。如果视频已过期或本地没有抽帧结果，
对应 txt 会被跳过；只有可用帧数不少于 `--min_num_images` 的 sequence 会进入评测。

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
