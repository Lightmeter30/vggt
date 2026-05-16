"""RealX3D 相机位姿评测模块。

评测 VGGT 在 RealX3D 退化图像上的相机位姿估计性能。
输出每个 condition/scene/split 的 AUC@5/10/20 及 per-frame 误差。
"""

import csv
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from vggt.utils.load_fn import load_and_preprocess_images_from_objects
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

ALL_CONDITIONS = [
    "defocus_mild", "defocus_strong", "motion_mild", "motion_strong",
    "dynamic", "reflection", "lowlight", "smoke", "varyexp",
]
ALL_SCENES = [
    "Akikaze", "BlueHawaii", "Chocolate", "Cupcake", "GearWorks", "Hinoki",
    "Koharu", "Laboratory", "Limon", "MilkCookie", "Natsume", "Popcorn",
    "Sculpture", "Shirohana", "Ujikintoki",
]
ALL_SPLITS = ["train", "val", "test"]


def add_arguments(parser):
    parser.add_argument("--data_root", type=str, required=True, help="RealX3D/data_4 根目录路径")
    parser.add_argument(
        "--conditions", type=str, nargs="+", default=None,
        help="要评测的 condition 列表，默认全部",
    )
    parser.add_argument(
        "--scenes", type=str, nargs="+", default=None,
        help="要评测的场景列表，默认全部",
    )
    parser.add_argument(
        "--splits", type=str, nargs="+", default=["train", "val"],
        help="要评测的数据划分，默认 train val",
    )
    parser.add_argument("--max_frames", type=int, default=None, help="每序列最多使用帧数，用于快速调试")
    parser.add_argument(
        "--output_dir", type=str, default="outputs/realx3d_vggt_pose",
        help="评测结果输出目录",
    )
    parser.add_argument(
        "--resize_long_side", type=int, default=None,
        help="预处理时限制图像长边最大尺寸",
    )
    parser.add_argument(
        "--pose_error_mode", type=str, default="max_rot_trans",
        choices=("max_rot_trans", "rotation_only"),
        help="位姿误差计算模式：max(R_err, t_err) 或仅旋转误差",
    )


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def _load_transforms(data_root, condition, scene, split):
    """读取 transforms_{split}.json，返回原始 dict。"""
    json_path = Path(data_root) / condition / scene / f"transforms_{split}.json"
    if not json_path.is_file():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _frame_name(frame):
    return Path(frame["file_path"]).name


def _select_frame_names_from_transforms(transforms, max_frames=None):
    frames = [
        frame for frame in transforms.get("frames", [])
        if "file_path" in frame and "transform_matrix" in frame
    ]
    if max_frames is not None:
        frames = frames[:max_frames]
    return [_frame_name(frame) for frame in frames]


def _filter_frames_by_names(frames, frame_names):
    frame_by_name = {_frame_name(frame): frame for frame in frames if "file_path" in frame}
    selected = []
    for name in frame_names:
        frame = frame_by_name.get(name)
        if frame is None:
            warnings.warn(f"未在 clean split 中找到与 train 对应的帧: {name}")
            continue
        selected.append(frame)
    return selected


def _load_realx3d_sequence(
    data_root,
    condition,
    scene,
    split,
    max_frames=None,
    resize_long_side=None,
    frame_names=None,
):
    """加载一个 {condition}/{scene}/{split} 的图像和 GT 位姿。

    RealX3D 的 transforms.json 使用 NeRF/Blender 坐标系，VGGT 使用 OpenCV 坐标系。
    对 c2w 矩阵需要同时转换 world basis 和 camera basis：
        T_cv = F @ T_blender @ F,  F = diag(1, -1, -1, 1)

    Returns:
        dict: {"image_objects": [...], "gt_c2w": np.ndarray (N,4,4), "file_names": [...]}
        或 None（数据缺失时）
    """
    transforms = _load_transforms(data_root, condition, scene, split)
    if transforms is None:
        return None

    scene_dir = Path(data_root) / condition / scene
    frames = list(transforms["frames"])
    if frame_names is not None:
        frames = _filter_frames_by_names(frames, frame_names)
    elif max_frames is not None:
        frames = frames[:max_frames]

    # Blender/NeRF → OpenCV 的坐标基变换。c2w 需要左右两侧都应用。
    blender_to_opencv = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)

    image_objects = []
    gt_c2w_list = []
    file_names = []

    for frame in frames:
        if "file_path" not in frame or "transform_matrix" not in frame:
            continue

        image_rel = frame["file_path"]
        image_path = scene_dir / image_rel
        if not image_path.is_file():
            warnings.warn(f"图像缺失: {image_path}")
            continue

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            warnings.warn(f"无法读取图像: {image_path}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if resize_long_side is not None:
            h, w = image_rgb.shape[:2]
            long_side = max(h, w)
            if long_side > resize_long_side:
                scale = resize_long_side / long_side
                new_w, new_h = int(w * scale), int(h * scale)
                image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        image_objects.append(image_rgb)

        c2w_blender = np.asarray(frame["transform_matrix"], dtype=np.float64)
        c2w_cv = blender_to_opencv @ c2w_blender @ blender_to_opencv
        gt_c2w_list.append(c2w_cv)
        file_names.append(image_rel)

    if len(image_objects) == 0:
        return None

    return {
        "image_objects": image_objects,
        "gt_c2w": np.stack(gt_c2w_list, axis=0),
        "file_names": file_names,
    }


def _build_realx3d_tasks(data_root, conditions, scenes, splits, max_frames=None):
    """构建 RealX3D 评测任务。

    当 split 正好为 train/val 时，train 作为退化 condition 评测；
    val 作为 paired clean 评测，输出目录使用 {condition}_clean，summary 中合并为 clean。
    """
    data_root = Path(data_root)
    paired_train_val = len(splits) == 2 and set(splits) == {"train", "val"}
    tasks = []
    skipped = []

    if paired_train_val:
        for condition in conditions:
            for scene in scenes:
                train_transforms = _load_transforms(data_root, condition, scene, "train")
                val_transforms = _load_transforms(data_root, condition, scene, "val")

                if train_transforms is None:
                    skipped.append(f"{condition}/{scene}/train")
                else:
                    tasks.append({
                        "data_condition": condition,
                        "output_condition": condition,
                        "condition": condition,
                        "scene": scene,
                        "split": "train",
                        "frame_names": None,
                    })

                if val_transforms is None:
                    skipped.append(f"{condition}/{scene}/val")
                elif train_transforms is None:
                    skipped.append(f"{condition}_clean/{scene}/val（缺少 train 配对帧）")
                else:
                    tasks.append({
                        "data_condition": condition,
                        "output_condition": f"{condition}_clean",
                        "condition": "clean",
                        "scene": scene,
                        "split": "val",
                        "frame_names": _select_frame_names_from_transforms(train_transforms, max_frames=max_frames),
                    })
        return tasks, skipped

    for condition in conditions:
        for scene in scenes:
            for split in splits:
                transforms_path = data_root / condition / scene / f"transforms_{split}.json"
                if transforms_path.is_file():
                    tasks.append({
                        "data_condition": condition,
                        "output_condition": condition,
                        "condition": condition,
                        "scene": scene,
                        "split": split,
                        "frame_names": None,
                    })
                else:
                    skipped.append(f"{condition}/{scene}/{split}")
    return tasks, skipped


# ---------------------------------------------------------------------------
# VGGT 推理
# ---------------------------------------------------------------------------

def _predict_camera_poses(model, image_objects, device, dtype):
    """VGGT 推理并返回 camera-to-world 位姿。

    VGGT 输出的外参是 OpenCV camera-from-world 3x4 矩阵，
    需要转换为 camera-to-world (c2w):
        R_c2w = R_cfW^T
        t_c2w = -R_cfW^T @ t_cfW
    """
    images = load_and_preprocess_images_from_objects(image_objects, mode="crop").to(device)

    with torch.no_grad():
        if device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
        else:
            predictions = model(images)

    pose_encoding = predictions["pose_enc"].to(torch.float64)
    # extrinsics: [B, S, 3, 4] camera-from-world [R|t]
    extrinsics_cfw, _ = pose_encoding_to_extri_intri(pose_encoding, images.shape[-2:])
    extrinsics_cfw = extrinsics_cfw[0].to(torch.float64)  # [S, 3, 4]

    # 转换为 camera-to-world
    R_cfw = extrinsics_cfw[:, :3, :3]  # [S, 3, 3]
    t_cfw = extrinsics_cfw[:, :3, 3]   # [S, 3]

    R_c2w = R_cfw.transpose(-2, -1)   # R_c2w = R_cfW^T
    t_c2w = -torch.bmm(R_c2w, t_cfw.unsqueeze(-1)).squeeze(-1)  # t_c2w = -R^T @ t

    pred_c2w = torch.zeros(extrinsics_cfw.size(0), 4, 4, dtype=torch.float64, device=device)
    pred_c2w[:, :3, :3] = R_c2w
    pred_c2w[:, :3, 3] = t_c2w
    pred_c2w[:, 3, 3] = 1.0

    # sanity checks
    for i in range(R_c2w.size(0)):
        det = float(torch.linalg.det(R_c2w[i]))
        if abs(det - 1.0) > 0.1:
            warnings.warn(f"帧 {i} 旋转矩阵 det={det:.4f}，偏离 1")

    return pred_c2w.cpu().numpy()


# ---------------------------------------------------------------------------
# 坐标系对齐
# ---------------------------------------------------------------------------

def _extract_camera_centers(c2w):
    """从 camera-to-world 矩阵提取相机中心在世界坐标系中的位置。

    c2w = [[R, t], [0,0,0,1]]，相机中心 C = t。
    """
    return c2w[..., :3, 3].copy()


def _c2w_rotation(c2w):
    """提取 c2w 矩阵中的旋转部分。"""
    return c2w[..., :3, :3].copy()


def _align_sim3(pred_c2w, gt_c2w):
    """Umeyama Sim(3) 对齐：将预测相机位姿对齐到 GT 坐标系。

    用预测相机中心对齐到 GT 相机中心，求解 scale s、rotation R、translation t，
    然后将预测的 rotation 和 center 变换到 GT 坐标系。

    Returns:
        aligned_c2w: ndarray (N, 4, 4)
        alignment: dict with keys s, R, t
    """
    C_pred = _extract_camera_centers(pred_c2w)
    C_gt = _extract_camera_centers(gt_c2w)
    R_pred = _c2w_rotation(pred_c2w)

    N = C_pred.shape[0]
    if N < 3:
        warnings.warn(f"帧数={N} < 3，Sim(3) 对齐可能不稳定")
        if N == 1:
            s, R_align, t_align = 1.0, np.eye(3), C_gt[0] - C_pred[0]
        else:
            s = 1.0
            mu_pred = C_pred.mean(axis=0)
            mu_gt = C_gt.mean(axis=0)
            C_pred_centered = C_pred - mu_pred
            C_gt_centered = C_gt - mu_gt
            H = C_pred_centered.T @ C_gt_centered
            U, _, Vt = np.linalg.svd(H)
            S_fix = np.eye(3)
            if np.linalg.det(U) * np.linalg.det(Vt) < 0:
                S_fix[2, 2] = -1
            R_align = Vt.T @ S_fix @ U.T
            t_align = mu_gt - s * R_align @ mu_pred
    else:
        # Umeyama 算法
        mu_pred = C_pred.mean(axis=0)
        mu_gt = C_gt.mean(axis=0)
        C_pred_centered = C_pred - mu_pred
        C_gt_centered = C_gt - mu_gt

        var_pred = np.sum(C_pred_centered ** 2) / N
        H = (C_pred_centered.T @ C_gt_centered) / N

        U, D, Vt = np.linalg.svd(H)

        # 处理反射情况
        S = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[2, 2] = -1

        R_align = Vt.T @ S @ U.T
        s = np.trace(np.diag(D) @ S) / var_pred if var_pred > 1e-12 else 1.0
        t_align = mu_gt - s * R_align @ mu_pred

    # 应用对齐
    R_aligned = np.empty_like(R_pred)
    C_aligned = np.empty_like(C_pred)
    for i in range(N):
        R_aligned[i] = R_align @ R_pred[i]
        C_aligned[i] = s * R_align @ C_pred[i] + t_align

    aligned_c2w = np.zeros_like(pred_c2w)
    aligned_c2w[:, :3, :3] = R_aligned
    aligned_c2w[:, :3, 3] = C_aligned
    aligned_c2w[:, 3, 3] = 1.0

    return aligned_c2w, {"s": float(s), "R": R_align.tolist(), "t": t_align.tolist()}


# ---------------------------------------------------------------------------
# 误差计算
# ---------------------------------------------------------------------------

def _rotation_error_deg(R_pred, R_gt):
    """计算旋转误差（度）。

    err_R = arccos((trace(R_pred^T @ R_gt) - 1) / 2)
    """
    trace_val = np.trace(R_pred.T @ R_gt)
    # clamp to [-1, 3] 避免数值误差
    cos_angle = (trace_val - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.arccos(cos_angle) * 180.0 / np.pi)


def _translation_angle_error_deg(C_pred, C_gt):
    """计算相机中心方向角误差（度）。

    使用 aligned camera center 与 GT camera center 的方向角误差；
    若范数过小，回退为中心 L2 误差归一化后的诊断项。
    """
    norm_pred = np.linalg.norm(C_pred)
    norm_gt = np.linalg.norm(C_gt)
    if norm_pred < 1e-8 or norm_gt < 1e-8:
        # 退化：用 L2 误差替代，记录 warning
        warnings.warn(
            f"相机中心范数过小 (pred={norm_pred:.2e}, gt={norm_gt:.2e})，回退为 L2 误差"
        )
        return float(np.linalg.norm(C_pred - C_gt))

    cos_angle = np.dot(C_pred, C_gt) / (norm_pred * norm_gt)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.arccos(cos_angle) * 180.0 / np.pi)


def _compute_pose_errors(aligned_c2w, gt_c2w, mode="max_rot_trans"):
    """计算逐帧位姿误差。

    Args:
        aligned_c2w: 对齐后的预测 c2w，shape (N, 4, 4)
        gt_c2w: GT c2w，shape (N, 4, 4)
        mode: "max_rot_trans" 或 "rotation_only"

    Returns:
        list[dict]: 每帧 {image_name, rot_err_deg, trans_err_deg, pose_err_deg}
    """
    N = aligned_c2w.shape[0]
    results = []
    for i in range(N):
        R_pred = aligned_c2w[i, :3, :3]
        C_pred = aligned_c2w[i, :3, 3]
        R_gt = gt_c2w[i, :3, :3]
        C_gt = gt_c2w[i, :3, 3]

        rot_err = _rotation_error_deg(R_pred, R_gt)
        trans_err = _translation_angle_error_deg(C_pred, C_gt)

        if mode == "rotation_only":
            pose_err = rot_err
        else:
            pose_err = max(rot_err, trans_err)

        results.append({
            "rot_err_deg": rot_err,
            "trans_err_deg": trans_err,
            "pose_err_deg": pose_err,
        })

    return results


def _compute_auc(errors, thresholds=(5, 10, 20)):
    """计算 pose error 在给定阈值下的 AUC。

    对每个阈值 T，绘制 recall-error 曲线：
    - x 轴：误差（度），范围 [0, T]
    - y 轴：recall = 误差 ≤ x 的帧数 / 总帧数
    - AUC = 曲线下面积 / T × 100

    Args:
        errors: 1D array of per-frame pose_err_deg
        thresholds: AUC 阈值列表

    Returns:
        dict: {f"auc{t}": float} 百分比 0-100
    """
    errors = np.asarray(errors, dtype=np.float64)
    total_frames = max(len(errors), 1)

    auc_results = {}
    for threshold in thresholds:
        e = np.sort(errors)
        e = e[e <= threshold]
        if len(e) == 0:
            auc_results[f"auc{threshold}"] = 0.0
            continue

        recall = np.arange(1, len(e) + 1) / total_frames
        # 曲线从 (0, 0) 开始，到 (threshold, recall[-1]) 结束
        e_ext = np.concatenate([[0.0], e, [float(threshold)]])
        r_ext = np.concatenate([[0.0], recall, [recall[-1]]])
        area = float(np.trapz(r_ext, e_ext))
        auc_results[f"auc{threshold}"] = area / float(threshold) * 100.0

    return auc_results


# ---------------------------------------------------------------------------
# 序列评测
# ---------------------------------------------------------------------------

def _evaluate_sequence(model, data_root, condition, scene, split, device, dtype,
                       max_frames=None, pose_error_mode="max_rot_trans",
                       resize_long_side=None, data_condition=None,
                       output_condition=None, frame_names=None):
    """评测单个 {condition}/{scene}/{split} 的相机位姿。

    Returns:
        dict: 包含 metrics、per_frame 结果、对齐参数等
    """
    seq_data = _load_realx3d_sequence(
        data_root, data_condition or condition, scene, split,
        max_frames=max_frames,
        resize_long_side=resize_long_side,
        frame_names=frame_names,
    )
    if seq_data is None or len(seq_data["image_objects"]) == 0:
        return None

    image_objects = seq_data["image_objects"]
    gt_c2w = seq_data["gt_c2w"]
    file_names = seq_data["file_names"]
    num_frames = len(image_objects)

    # VGGT 推理
    pred_c2w = _predict_camera_poses(model, image_objects, device, dtype)

    # Sim(3) 对齐
    aligned_c2w, alignment = _align_sim3(pred_c2w, gt_c2w)

    # 计算逐帧误差
    per_frame = _compute_pose_errors(aligned_c2w, gt_c2w, mode=pose_error_mode)

    # 添加文件名
    for i, name in enumerate(file_names):
        per_frame[i]["image_name"] = name

    # 汇总指标
    rot_errs = [f["rot_err_deg"] for f in per_frame]
    trans_errs = [f["trans_err_deg"] for f in per_frame]
    pose_errs = [f["pose_err_deg"] for f in per_frame]
    auc = _compute_auc(pose_errs)

    return {
        "condition": condition,
        "data_condition": data_condition or condition,
        "output_condition": output_condition or condition,
        "scene": scene,
        "split": split,
        "num_frames": num_frames,
        "per_frame": per_frame,
        "rot_errs": np.array(rot_errs),
        "trans_errs": np.array(trans_errs),
        "pose_errs": np.array(pose_errs),
        "auc": auc,
        "pred_c2w_raw": pred_c2w,
        "pred_c2w_aligned": aligned_c2w,
        "gt_c2w": gt_c2w,
        "alignment": alignment,
    }


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def _write_sequence_outputs(results, output_dir):
    """写出单个序列的 per_frame_errors.csv、metrics.json、.npy 文件和 config.json。"""
    out_dir = Path(output_dir) / results.get("output_condition", results["condition"]) / results["scene"] / results["split"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # per_frame_errors.csv
    csv_path = out_dir / "per_frame_errors.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "rot_err_deg", "trans_err_deg", "pose_err_deg"])
        writer.writeheader()
        writer.writerows(results["per_frame"])

    # metrics.json
    rot = results["rot_errs"]
    trans = results["trans_errs"]
    pose = results["pose_errs"]
    metrics = {
        "num_frames": int(results["num_frames"]),
        "mean_rot_err": float(np.mean(rot)),
        "median_rot_err": float(np.median(rot)),
        "mean_trans_err": float(np.mean(trans)),
        "median_trans_err": float(np.median(trans)),
        "mean_pose_err": float(np.mean(pose)),
        "median_pose_err": float(np.median(pose)),
        **results["auc"],
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 位姿数组
    np.save(out_dir / "pred_poses_aligned.npy", results["pred_c2w_aligned"])
    np.save(out_dir / "gt_poses.npy", results["gt_c2w"])

    # config.json
    config = {
        "alignment": results["alignment"],
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return metrics


def _write_summary_csvs(all_results, output_dir):
    """写出 summary.csv、summary_by_condition.csv、summary_by_split.csv。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in all_results:
        rot = r["rot_errs"]
        trans = r["trans_errs"]
        pose = r["pose_errs"]
        rows.append({
            "condition": r["condition"],
            "scene": r["scene"],
            "split": r["split"],
            "num_frames": int(r["num_frames"]),
            **{f"auc{k}": r["auc"][f"auc{k}"] for k in [5, 10, 20]},
            "mean_rot_err": float(np.mean(rot)),
            "median_rot_err": float(np.median(rot)),
            "mean_trans_err": float(np.mean(trans)),
            "median_trans_err": float(np.median(trans)),
            "mean_pose_err": float(np.mean(pose)),
            "median_pose_err": float(np.median(pose)),
        })

    fieldnames = [
        "condition", "scene", "split", "num_frames",
        "auc5", "auc10", "auc20",
        "mean_rot_err", "median_rot_err", "mean_trans_err", "median_trans_err",
        "mean_pose_err", "median_pose_err",
    ]

    # summary.csv - 全量明细
    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # summary_by_condition.csv
    by_condition = {}
    for row in rows:
        cond = row["condition"]
        if cond not in by_condition:
            by_condition[cond] = {k: [] for k in fieldnames[3:]}
        for k in fieldnames[3:]:
            by_condition[cond][k].append(row[k])

    cond_rows = []
    for cond, vals in sorted(by_condition.items()):
        r = {"condition": cond, "num_scenes": len(vals["num_frames"])}
        for k in fieldnames[3:]:
            r[k] = float(np.mean(vals[k]))
        cond_rows.append(r)

    cond_fields = ["condition", "num_scenes"] + fieldnames[3:]
    with open(output_dir / "summary_by_condition.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cond_fields)
        writer.writeheader()
        writer.writerows(cond_rows)

    # summary_by_split.csv
    by_split = {}
    for row in rows:
        sp = row["split"]
        if sp not in by_split:
            by_split[sp] = {k: [] for k in fieldnames[3:]}
        for k in fieldnames[3:]:
            by_split[sp][k].append(row[k])

    split_rows = []
    for sp, vals in sorted(by_split.items()):
        r = {"split": sp, "num_scenes": len(vals["num_frames"])}
        for k in fieldnames[3:]:
            r[k] = float(np.mean(vals[k]))
        split_rows.append(r)

    split_fields = ["split", "num_scenes"] + fieldnames[3:]
    with open(output_dir / "summary_by_split.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=split_fields)
        writer.writeheader()
        writer.writerows(split_rows)


# ---------------------------------------------------------------------------
# 顶层入口
# ---------------------------------------------------------------------------

def run(args, model, device, dtype):
    conditions = args.conditions if args.conditions else ALL_CONDITIONS
    scenes = args.scenes if args.scenes else ALL_SCENES
    splits = args.splits if args.splits else ALL_SPLITS

    data_root = Path(args.data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"RealX3D 数据根目录不存在: {data_root}")

    # 检查哪些 condition/scene/split 实际存在
    tasks, skipped = _build_realx3d_tasks(
        data_root=data_root,
        conditions=conditions,
        scenes=scenes,
        splits=splits,
        max_frames=args.max_frames,
    )

    if skipped:
        print(f"跳过 {len(skipped)} 个缺失的 condition/scene/split 组合")
        for s in skipped[:10]:
            print(f"  缺失: {s}")
        if len(skipped) > 10:
            print(f"  ... 及其他 {len(skipped) - 10} 个")

    if not tasks:
        raise RuntimeError(f"未找到任何有效的 transforms 文件，请检查 --data_root: {data_root}")

    print(f"共 {len(tasks)} 个评测任务")

    all_results = []
    for task in tqdm(tasks, desc="RealX3D 评测"):
        condition = task["condition"]
        scene = task["scene"]
        split = task["split"]
        try:
            seq_result = _evaluate_sequence(
                model=model,
                data_root=data_root,
                condition=condition,
                scene=scene,
                split=split,
                device=device,
                dtype=dtype,
                max_frames=args.max_frames,
                pose_error_mode=args.pose_error_mode,
                resize_long_side=args.resize_long_side,
                data_condition=task["data_condition"],
                output_condition=task["output_condition"],
                frame_names=task["frame_names"],
            )
        except Exception as exc:
            print(f"评测失败 [{task['output_condition']}/{scene}/{split}]: {exc}")
            continue

        if seq_result is None:
            print(f"跳过（无有效数据）: {task['output_condition']}/{scene}/{split}")
            continue

        # 写出单序列结果
        metrics = _write_sequence_outputs(seq_result, args.output_dir)
        all_results.append(seq_result)

        # 简要输出
        auc_str = ", ".join(f"auc{k}={metrics[f'auc{k}']:.1f}" for k in [5, 10, 20])
        print(
            f"[{task['output_condition']}/{scene}/{split}] frames={seq_result['num_frames']} "
            f"mean_pose_err={metrics['mean_pose_err']:.2f}°  {auc_str}"
        )

    if not all_results:
        print("警告: 没有成功评测的序列")
        return []

    # 写出汇总 CSV
    _write_summary_csvs(all_results, args.output_dir)

    # 打印总体摘要
    all_rot = np.concatenate([r["rot_errs"] for r in all_results])
    all_trans = np.concatenate([r["trans_errs"] for r in all_results])
    all_pose = np.concatenate([r["pose_errs"] for r in all_results])

    print(f"\n===== RealX3D 评测摘要 =====")
    print(f"成功评测序列数: {len(all_results)}")
    print(f"总帧数: {len(all_pose)}")
    print(f"mean_rot_err: {np.mean(all_rot):.2f}°")
    print(f"median_rot_err: {np.median(all_rot):.2f}°")
    print(f"mean_trans_err: {np.mean(all_trans):.2f}°")
    print(f"median_trans_err: {np.median(all_trans):.2f}°")
    print(f"mean_pose_err: {np.mean(all_pose):.2f}°")
    print(f"median_pose_err: {np.median(all_pose):.2f}°")
    for k in [5, 10, 20]:
        print(f"auc{k}: {_compute_auc(all_pose, thresholds=[k])[f'auc{k}']:.1f}")
    print(f"结果已保存到: {args.output_dir}")

    return all_results
