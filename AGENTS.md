# AGENTS.md

本文件面向在本仓库内工作的代码代理，目标是让代理在尽量少的往返下完成可落地的修改、排查和验证。

## 1. 项目是什么

这是一个以 `VGGT` 为核心的 Python / PyTorch 项目，提供：

- 多视图或单视图 3D 几何推理
- Gradio / Viser 可视化
- 导出到 COLMAP 格式
- 基于 `training/` 的微调与训练流程

核心模型定义在 [vggt/models/vggt.py](/home/zwr/code/my-vggt/vggt/models/vggt.py)，由一个共享 `Aggregator` 和多个任务头组成：

- `camera_head`: 相机位姿与内参编码
- `depth_head`: 深度图与置信度
- `point_head`: 点图与置信度
- `track_head`: 点轨迹、可见性、置信度

## 2. 信息优先级

当文档与代码不一致时，按以下顺序判断：

1. 源码本身
2. [README.md](/home/zwr/code/my-vggt/README.md)
3. [training/README.md](/home/zwr/code/my-vggt/training/README.md)
4. [docs/package.md](/home/zwr/code/my-vggt/docs/package.md)
5. `my_docs/` 下的中文分析文档

说明：`my_docs/` 更像本地阅读笔记和源码分析，适合辅助理解，但不应覆盖源码事实。

## 3. 目录地图

- [vggt/](/home/zwr/code/my-vggt/vggt): 核心包
- [vggt/models/](/home/zwr/code/my-vggt/vggt/models): 主模型与聚合器
- [vggt/heads/](/home/zwr/code/my-vggt/vggt/heads): camera / depth / point / track heads
- [vggt/layers/](/home/zwr/code/my-vggt/vggt/layers): Transformer、RoPE、patch embedding 等底层模块
- [vggt/utils/](/home/zwr/code/my-vggt/vggt/utils): 几何、姿态、图像加载、可视化辅助
- [vggt/dependency/](/home/zwr/code/my-vggt/vggt/dependency): COLMAP、追踪、失真处理等依赖逻辑
- [training/](/home/zwr/code/my-vggt/training): 训练系统
- [training/config/](/home/zwr/code/my-vggt/training/config): Hydra 配置
- [training/data/](/home/zwr/code/my-vggt/training/data): Dataset / DataLoader / 预处理
- [training/train_utils/](/home/zwr/code/my-vggt/training/train_utils): checkpoint、optimizer、logging、DDP 工具
- [examples/](/home/zwr/code/my-vggt/examples): 示例图片与视频
- [visual_util.py](/home/zwr/code/my-vggt/visual_util.py): GLB 构建、天空分割下载、可视化辅助
- [demo_gradio.py](/home/zwr/code/my-vggt/demo_gradio.py): Web Demo
- [demo_viser.py](/home/zwr/code/my-vggt/demo_viser.py): Viser 3D 浏览器
- [demo_colmap.py](/home/zwr/code/my-vggt/demo_colmap.py): COLMAP 导出

## 4. 环境与依赖

### 基础推理依赖

项目不会在包元数据中自动安装 `torch` / `torchvision`，这是有意为之，以避免 CUDA 版本冲突。常规顺序是：

```bash
pip install torch torchvision --index-url <与你环境匹配的源>
pip install -e .
```

或使用：

```bash
pip install -r requirements.txt
```

### Demo / 可视化依赖

```bash
pip install -r requirements_demo.txt
```

这会引入 `gradio`、`viser`、`opencv-python`、`onnxruntime`、`trimesh` 等可视化相关依赖。

### 训练依赖注意事项

训练代码除了 PyTorch 外，还会用到：

- `hydra-core`
- `omegaconf`
- `fvcore`
- `iopath`
- `tensorboard`

其中一部分没有集中写在基础 `requirements.txt` 中。若训练脚本报缺包，优先补齐上述依赖。

## 5. 常用工作流

### 5.1 本地导入与快速推理

最小使用路径来自 [README.md](/home/zwr/code/my-vggt/README.md)：

```python
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
```

模型权重通常通过 Hugging Face 自动下载。多数 demo 默认需要联网首次拉取权重。

### 5.2 Gradio Demo

```bash
python demo_gradio.py
```

说明：

- 默认会加载远程权重
- 需要 CUDA；脚本里对无 CUDA 情况会直接报错
- 上传文件后会生成 `input_images_*` 目录和中间产物

### 5.3 Viser 可视化

```bash
python demo_viser.py --image_folder examples/kitchen/images
```

常用参数：

- `--use_point_map`
- `--mask_sky`
- `--port 8080`

### 5.4 导出 COLMAP

```bash
python demo_colmap.py --scene_dir examples/kitchen
python demo_colmap.py --scene_dir examples/kitchen --use_ba
```

要求：

- `scene_dir/images/` 下应只放图片
- BA 路径依赖 `pycolmap`，以及追踪相关依赖

输出通常写到：

- `scene_dir/sparse/`
- `scene_dir/sparse/points.ply`

### 5.5 训练 / 微调

训练入口不是仓库根目录，而是 `training/` 目录内的 [launch.py](/home/zwr/code/my-vggt/training/launch.py)。运行前先检查：

- [training/config/default.yaml](/home/zwr/code/my-vggt/training/config/default.yaml)
- `CO3D_DIR`
- `CO3D_ANNOTATION_DIR`
- `checkpoint.resume_checkpoint_path`

常用命令：

```bash
cd training
torchrun --nproc_per_node=4 launch.py
```

注意：

- `launch.py` 通过相对导入 `trainer` 和相对 `config/` 路径工作，因此默认应在 `training/` 目录内执行
- 默认配置是微调模式：开启 camera/depth，关闭 point/track，并冻结 `aggregator`

## 6. 关键源码约束

### 6.1 输入输出形状

- `VGGT.forward()` 支持 `[S, 3, H, W]` 或 `[B, S, 3, H, W]`
- 如果引入新逻辑，不要破坏当前两种输入形式
- Demo 和工具函数广泛依赖这一约定

### 6.2 坐标系约定

训练与导出代码默认使用 OpenCV `camera-from-world` 约定。涉及以下位置时必须保持一致：

- 相机外参处理
- 深度反投影
- COLMAP 导出
- 自定义数据集接入

不要在未明确说明的情况下混入 OpenGL / PyTorch3D 约定。

### 6.3 图像预处理约定

[vggt/utils/load_fn.py](/home/zwr/code/my-vggt/vggt/utils/load_fn.py) 是推理图像预处理的统一入口：

- `load_and_preprocess_images()`: 常规推理
- `load_and_preprocess_images_square()`: COLMAP 导出路径常用

如果修改预处理逻辑，需要同步考虑：

- patch size 对齐
- 不同尺寸图片的 padding / crop 行为
- alpha 通道转白底

### 6.4 Demo 对网络与文件系统有副作用

以下脚本会下载模型、写本地输出或下载外部辅助模型：

- [demo_gradio.py](/home/zwr/code/my-vggt/demo_gradio.py)
- [demo_viser.py](/home/zwr/code/my-vggt/demo_viser.py)
- [demo_colmap.py](/home/zwr/code/my-vggt/demo_colmap.py)
- [visual_util.py](/home/zwr/code/my-vggt/visual_util.py)

例如：

- 首次运行会下载 `facebook/VGGT-1B` 权重
- 开启天空分割时可能下载 `skyseg.onnx`
- Gradio 会写 `input_images_*`

修改这些脚本前，要先确认是逻辑缺陷、CLI 需求还是仅仅是本地产物带来的噪音。

## 7. 不应随意改动的内容

- [examples/](/home/zwr/code/my-vggt/examples): 示例素材，默认视作只读
- `input_images_*`: 通常是用户本地推理输出，除非用户明确要求，不要删除
- `.gradio/`: 本地运行产物
- `my_docs/`: 本地分析文档，可补充，但不应当作运行时依赖

仓库当前可能存在未跟踪文件和本地产物，修改前先看 `git status`，避免把用户生成的数据误当成待清理文件。

## 8. 修改建议

### 8.1 改模型逻辑时

优先阅读：

- [vggt/models/vggt.py](/home/zwr/code/my-vggt/vggt/models/vggt.py)
- [vggt/models/aggregator.py](/home/zwr/code/my-vggt/vggt/models/aggregator.py)
- 对应 head 文件

如果改 `Aggregator` 或 head，通常还要同步检查：

- `forward()` 输出键名
- demo 中对 `predictions` 的消费
- 训练 loss 侧是否仍然兼容

### 8.2 改训练逻辑时

优先阅读：

- [training/trainer.py](/home/zwr/code/my-vggt/training/trainer.py)
- [training/loss.py](/home/zwr/code/my-vggt/training/loss.py)
- [training/config/default.yaml](/home/zwr/code/my-vggt/training/config/default.yaml)
- [training/data/datasets/co3d.py](/home/zwr/code/my-vggt/training/data/datasets/co3d.py)

Hydra 配置改动要特别注意默认值是否仍能表达当前训练策略。

### 8.3 改可视化 / 导出逻辑时

优先阅读：

- [visual_util.py](/home/zwr/code/my-vggt/visual_util.py)
- [demo_viser.py](/home/zwr/code/my-vggt/demo_viser.py)
- [demo_colmap.py](/home/zwr/code/my-vggt/demo_colmap.py)

尤其注意：

- confidence threshold 的定义是百分位还是绝对值
- 导出路径是否保持向后兼容
- 是否引入额外网络下载或重计算

## 9. 验证策略

本仓库当前没有现成的 `pytest` 测试套件。完成修改后，优先做针对性验证：

### 最低限度

```bash
python -m py_compile <你改过的 Python 文件>
```

### 推理相关改动

- 至少验证 import 不报错
- 如环境允许，使用 `examples/kitchen/images` 跑一次对应 demo

### 训练相关改动

- 至少验证配置与导入路径没有明显错误
- 若无法实际开训，明确说明缺失的依赖、数据集或 GPU 条件

### Demo 相关改动

- 明确是否依赖联网下载权重
- 明确是否依赖 CUDA
- 明确是否会产生本地输出目录

## 10. 代理工作准则

- 优先小范围修改，不要为了风格统一做大面积无关重构
- 除非用户明确要求，不要清理示例数据、输出目录或未跟踪文件
- 修改训练配置时，保留需要用户替换的绝对路径占位项
- 任何涉及相机、坐标系、点云导出的修改，都要先确认数学约定
- 如果验证受限于网络、权重下载、缺少 GPU 或缺少数据集，要在结果中明确写出

## 11. 建议的最小排查顺序

1. 看 `README.md` 确认目标工作流
2. 看对应入口脚本或核心模块
3. 看配置文件和依赖是否齐全
4. 做最小可行修改
5. 运行 `py_compile` 或最小 smoke test
6. 明确剩余风险
