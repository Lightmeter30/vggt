# Local Fine-tune Notes

这份说明对应当前机器上已经整理好的本地 Co3D 子集微调流程。

## 固定路径

- Co3D 数据集: `/home/zwr/code/my-vggt/dataset/co3d`
- 本地生成 annotation: `/home/zwr/code/my-vggt/dataset/co3d-anno-local`
- 微调初始权重: `/home/zwr/code/my-vggt/ckpt/model.pt`
- 训练配置: [local_finetune.yaml](/home/zwr/code/my-vggt/training/config/local_finetune.yaml)

## 一键启动

在仓库根目录执行：

```bash
bash training/run_local_finetune.sh
```

默认行为：

- 使用 `my_vggt_relocation` conda 环境
- 使用 `GPU 0`
- 使用 `master_port=29601`
- 启动前自动重建一次 `dataset/co3d-anno-local`

## 指定 GPU

单卡：

```bash
bash training/run_local_finetune.sh 0
```

多卡：

```bash
bash training/run_local_finetune.sh 0,1
```

上面这条命令会自动把 `nproc_per_node` 设成 GPU 个数。

## 指定端口

```bash
bash training/run_local_finetune.sh 0 29611
```

如果默认 `29500` 或 `29601` 被占用，直接换一个端口即可。

## 显式指定进程数

```bash
bash training/run_local_finetune.sh 0,1 29611 2
```

## 常用环境变量

改 conda 环境：

```bash
CONDA_ENV_NAME=my_vggt_relocation bash training/run_local_finetune.sh 0
```

跳过 annotation 重建：

```bash
PREPARE_LOCAL_ANNO=0 bash training/run_local_finetune.sh 0
```

## 日志与输出

- 训练日志: `/home/zwr/code/my-vggt/training/logs/log.txt`
- TensorBoard: `/home/zwr/code/my-vggt/training/logs/tensorboard/`
- checkpoint: `/home/zwr/code/my-vggt/training/logs/local_co3d_finetune_localanno/ckpts/`

## 停止训练

如果是通过当前脚本启动的，可以停掉：

```bash
pkill -f 'torchrun --master_port 29601'
```

更稳妥的方式是先用 `ps -ef | grep torchrun` 找到对应进程，再按 PID 停止。
