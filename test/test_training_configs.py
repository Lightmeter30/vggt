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


def test_euroc_imu_film_uses_small_smoke_training_shape():
    with initialize_config_dir(
        version_base=None,
        config_dir=str(TRAINING_CONFIG_DIR),
    ):
        cfg = compose(config_name="euroc_imu_film")

    resolved = OmegaConf.to_container(cfg, resolve=True)

    assert resolved["img_size"] == 224
    assert resolved["max_img_per_gpu"] == 2
    assert resolved["data"]["train"]["common_config"]["img_nums"] == [2, 2]
    assert resolved["data"]["val"]["common_config"]["img_nums"] == [2, 2]
    assert resolved["data"]["val"]["dataset"]["dataset_configs"][0]["len_test"] == 3
