from evaluation.datasets.co3d import camera_pose as co3d_camera_pose
from evaluation.datasets.euroc import camera_pose as euroc_camera_pose
from evaluation.datasets.realestate10k import camera_pose as realestate10k_camera_pose


TASK_REGISTRY = {
    ("co3d", "camera_pose"): co3d_camera_pose,
    ("euroc", "camera_pose"): euroc_camera_pose,
    ("realestate10k", "camera_pose"): realestate10k_camera_pose,
}


def get_evaluator(dataset, task):
    key = (dataset, task)
    if key not in TASK_REGISTRY:
        supported = ", ".join(
            f"{registered_dataset}/{registered_task}"
            for registered_dataset, registered_task in sorted(TASK_REGISTRY.keys())
        )
        raise ValueError(
            f"Unsupported evaluation target: dataset={dataset}, task={task}. "
            f"Supported targets: {supported}"
        )
    return TASK_REGISTRY[key]
