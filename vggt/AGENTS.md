# AGENTS.md

## 范围

- `models/`: `VGGT` 主模型与 `Aggregator`。
- `heads/`: camera、depth、point、track heads。
- `layers/`: Transformer、RoPE、patch embedding 等底层模块。
- `utils/`: 几何、姿态、图像加载、可视化辅助。
- `dependency/`: COLMAP、追踪、失真处理等依赖逻辑。

## 核心入口

- `models/vggt.py`: `VGGT` 定义。
- `models/aggregator.py`: 共享特征聚合器。
- `models/imu_encoder.py`: 将每帧 IMU 时间窗编码为 motion token 和 motion risk。
- `models/visual_imu_fusion.py`: 使用 FiLM 调制视觉 patch tokens。
- `utils/load_fn.py`: 推理图像预处理统一入口。
- `utils/pose_enc.py`: 相机编码与转换。
- `utils/geometry.py`: 几何计算。

## 模型约定

- `VGGT.forward()` 输入支持 `[S, 3, H, W]` 和 `[B, S, 3, H, W]`。
- `VGGT.forward()` 可选输入 `imu_windows`、`imu_window_masks`、`degradation_metadata`；启用 `imu` 配置时必须提供 `imu_windows`。
- 输出键名需要兼容 demo、evaluation、training。
- 启用 IMU 时输出会额外包含 `motion_tokens`、`motion_risk`，训练侧会记录 `motion_risk_mean/std`。
- `aggregator(images, motion_tokens=None, imu_fusion=None)` 输出 `aggregated_tokens_list, ps_idx`。
- `Aggregator` 的 FiLM 调制只作用于 patch tokens，不能改变 camera/register special tokens。
- `VisualIMUFiLM` 最后一层零初始化，初始状态应等价于不调制视觉 token。
- `camera_head` 输出 `pose_enc`，再转换为 extrinsic `[3x4]`、intrinsic `[3x3]`。
- `depth_head` 输出 `depth_map, depth_conf`。
- `point_head` 输出 `point_map, point_conf`。
- `track_head` 输出 `track, vis_score, conf_score`。

## 坐标与图像约定

- 默认 OpenCV `camera-from-world`。
- 不要无说明混入 OpenGL / PyTorch3D 坐标系。
- 修改深度反投影、外参、点云、COLMAP 导出前先核对坐标方向。
- 预处理修改必须兼容 `load_and_preprocess_images()`。
- 评测路径常用 `load_and_preprocess_images_from_objects()`。
- COLMAP 路径常用 `load_and_preprocess_images_square()`。
- 尺寸处理需保持 patch size 对齐；当前关键尺寸需能被 14 整除。
- alpha 通道按现有逻辑转白底。

## 验证

```bash
conda run -n my_vggt_relocation python -m py_compile vggt/models/vggt.py
conda run -n my_vggt_relocation python -m py_compile vggt/models/aggregator.py
conda run -n my_vggt_relocation python -m py_compile vggt/models/imu_encoder.py
conda run -n my_vggt_relocation python -m py_compile vggt/models/visual_imu_fusion.py
conda run -n my_vggt_relocation python -m pytest test/test_imu_encoder.py test/test_visual_imu_fusion.py test/test_vggt_imu_forward.py -q
conda run -n my_vggt_relocation python -m py_compile vggt/utils/load_fn.py
conda run -n my_vggt_relocation python -m pytest test/ -q
```

## 规则

- 改 `Aggregator` 或 heads 后，同步检查 demo、evaluation、training 的消费路径。
- 改 IMU 编码或 FiLM 融合后，同步检查 `VGGT.forward()` 的无 IMU 兼容路径，确保 demo/evaluation 不需要额外参数也能继续调用。
- `attention_bias`、`degradation_reweight` 当前只是模型构造参数占位，未接入 forward 行为；不要在文档或实验记录里当作已实现能力描述。
- 推理 smoke test 可能依赖 CUDA、权重、本地图片和网络下载。
- 不要为了风格统一大面积重写 layers 或几何工具。
