from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_CONFIG_DIR = REPO_ROOT / "training" / "config"


def test_euroc_imu_film_gradient_clip_covers_imu_modules():
    with initialize_config_dir(
        version_base=None,
        config_dir=str(TRAINING_CONFIG_DIR),
    ):
        cfg = compose(config_name="euroc_imu_film")

    resolved = OmegaConf.to_container(cfg, resolve=True)
    module_names = []
    for clip_config in resolved["optim"]["gradient_clip"]["configs"]:
        names = clip_config["module_name"]
        module_names.extend(names if isinstance(names, list) else [names])

    assert "imu_encoder" in module_names
    assert "imu_fusion" in module_names


def test_euroc_imu_film_uses_stable_full_training_shape():
    with initialize_config_dir(
        version_base=None,
        config_dir=str(TRAINING_CONFIG_DIR),
    ):
        cfg = compose(config_name="euroc_imu_film")

    resolved = OmegaConf.to_container(cfg, resolve=True)

    assert resolved["exp_name"] == "euroc_imu_film_full_518_s6_mipg12_lr1e-4_new"
    assert resolved["img_size"] == 518
    assert resolved["max_img_per_gpu"] == 12
    assert resolved["max_epochs"] == 10
    assert resolved["val_epoch_freq"] == 5
    assert resolved["limit_train_batches"] == 800
    assert resolved["limit_val_batches"] == 400
    assert resolved["optim"]["optimizer"]["lr"] == 1e-4
    assert resolved["data"]["train"]["common_config"]["img_nums"] == [2, 12]
    assert resolved["data"]["val"]["common_config"]["img_nums"] == [6, 6]
    assert "len_test" not in resolved["data"]["val"]["dataset"]["dataset_configs"][0]


def test_euroc_imu_film_uses_asl_dataset_sequence_splits_without_degradation():
    with initialize_config_dir(
        version_base=None,
        config_dir=str(TRAINING_CONFIG_DIR),
    ):
        cfg = compose(config_name="euroc_imu_film")

    resolved = OmegaConf.to_container(cfg, resolve=True)

    for dataset_name in ("euroc", "tum_vi", "kaist_vi", "uma_vi"):
        assert dataset_name in resolved["vi_datasets"]
        assert dataset_name in resolved["sequence_splits"]["train"]
        assert dataset_name in resolved["sequence_splits"]["val"]

    train_configs = resolved["data"]["train"]["dataset"]["dataset_configs"]
    val_configs = resolved["data"]["val"]["dataset"]["dataset_configs"]

    assert all(config["_target_"] == "data.datasets.asl.ASLDataset" for config in train_configs)
    assert all(config["_target_"] == "data.datasets.asl.ASLDataset" for config in val_configs)
    assert all("degradation" not in config for config in train_configs + val_configs)
    assert train_configs[0]["sequence_names"] == resolved["sequence_splits"]["train"]["euroc"]
    assert val_configs[0]["sequence_names"] == resolved["sequence_splits"]["val"]["euroc"]
