import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


SEEN_CATEGORIES = [
    "apple",
    "backpack",
    "banana",
    "baseballbat",
    "baseballglove",
    "bench",
    "bicycle",
    "bottle",
    "bowl",
    "broccoli",
    "cake",
    "car",
    "carrot",
    "cellphone",
    "chair",
    "cup",
    "donut",
    "hairdryer",
    "handbag",
    "hydrant",
    "keyboard",
    "laptop",
    "microwave",
    "motorcycle",
    "mouse",
    "orange",
    "parkingmeter",
    "pizza",
    "plant",
    "stopsign",
    "teddybear",
    "toaster",
    "toilet",
    "toybus",
    "toyplane",
    "toytrain",
    "toytruck",
    "tv",
    "umbrella",
    "vase",
    "wineglass",
]


def viewpoint_to_opencv(frame_anno):
    viewpoint = frame_anno["viewpoint"]
    # Co3D frame_annotations store image size as [H, W].
    image_h, image_w = frame_anno["image"]["size"]

    # Co3D stores PyTorch3D camera extrinsics: +X left, +Y up, +Z forward.
    # The training code expects OpenCV camera-from-world: +X right, +Y down, +Z forward.
    mirror_xy = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
    R_p3d = np.asarray(viewpoint["R"], dtype=np.float32)
    T_p3d = np.asarray(viewpoint["T"], dtype=np.float32)
    extri_opencv = np.concatenate(
        [mirror_xy @ R_p3d.T, (mirror_xy @ T_p3d[:, None])],
        axis=1,
    )

    # Co3D stores NDC-isotropic intrinsics in PyTorch3D convention.
    # Convert them to OpenCV screen-space intrinsics in pixels.
    focal_ndc = np.asarray(viewpoint["focal_length"], dtype=np.float32)
    pp_ndc = np.asarray(viewpoint["principal_point"], dtype=np.float32)
    ndc_scale = min(image_h, image_w) / 2.0

    intri_opencv = np.eye(3, dtype=np.float32)
    intri_opencv[0, 0] = focal_ndc[0] * ndc_scale
    intri_opencv[1, 1] = focal_ndc[1] * ndc_scale
    intri_opencv[0, 2] = image_w / 2.0 - pp_ndc[0] * ndc_scale
    intri_opencv[1, 2] = image_h / 2.0 - pp_ndc[1] * ndc_scale

    return extri_opencv.tolist(), intri_opencv.tolist()


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_jgz(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def dump_jgz(path, payload):
    with gzip.open(path, "wt") as f:
        json.dump(payload, f)


def build_category_annotations(category_dir: Path):
    frame_annos = load_jgz(category_dir / "frame_annotations.jgz")

    frame_by_path = {frame["image"]["path"]: frame for frame in frame_annos}
    split_paths = {"train": set(), "test": set()}

    for set_list_path in sorted((category_dir / "set_lists").glob("set_lists_*.json")):
        payload = load_json(set_list_path)
        for seq_name, frame_no, image_path in payload.get("train", []):
            split_paths["train"].add(image_path)
        for key in ("val", "test"):
            for seq_name, frame_no, image_path in payload.get(key, []):
                split_paths["test"].add(image_path)

    outputs = {"train": defaultdict(list), "test": defaultdict(list)}
    stats = {
        "train_total": 0,
        "test_total": 0,
        "train_kept": 0,
        "test_kept": 0,
        "missing": 0,
    }

    root = category_dir.parent
    for split_name, selected_paths in split_paths.items():
        stats[f"{split_name}_total"] = len(selected_paths)

        for image_path in sorted(selected_paths):
            frame = frame_by_path.get(image_path)
            if frame is None:
                continue

            image_abs = root / image_path
            depth_abs = root / frame["depth"]["path"]
            mask_abs = root / frame["depth"]["mask_path"]
            if not image_abs.is_file() or not depth_abs.is_file() or not mask_abs.is_file():
                stats["missing"] += 1
                continue

            extri_opencv, intri_opencv = viewpoint_to_opencv(frame)
            outputs[split_name][frame["sequence_name"]].append(
                {
                    "filepath": image_path,
                    "extri": extri_opencv,
                    "intri": intri_opencv,
                    "frame_number": frame["frame_number"],
                }
            )
            stats[f"{split_name}_kept"] += 1

    final_outputs = {}
    for split_name, split_payload in outputs.items():
        final_outputs[split_name] = {}
        for seq_name, seq_items in split_payload.items():
            seq_items = sorted(seq_items, key=lambda x: x["frame_number"])
            for item in seq_items:
                item.pop("frame_number", None)
            final_outputs[split_name][seq_name] = seq_items

    return final_outputs, stats


def main():
    parser = argparse.ArgumentParser(description="Generate local Co3D annotations compatible with VGGT training.")
    parser.add_argument("--co3d_dir", required=True, help="Path to local Co3D root.")
    parser.add_argument("--output_dir", required=True, help="Directory to write generated *.jgz files.")
    args = parser.parse_args()

    co3d_dir = Path(args.co3d_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for category in SEEN_CATEGORIES:
        category_dir = co3d_dir / category
        train_payload = {}
        test_payload = {}

        if category_dir.is_dir():
            outputs, stats = build_category_annotations(category_dir)
            train_payload = outputs["train"]
            test_payload = outputs["test"]
            summary[category] = {
                **stats,
                "train_sequences": len(train_payload),
                "test_sequences": len(test_payload),
            }
        else:
            summary[category] = {
                "train_total": 0,
                "test_total": 0,
                "train_kept": 0,
                "test_kept": 0,
                "missing": 0,
                "train_sequences": 0,
                "test_sequences": 0,
            }

        dump_jgz(output_dir / f"{category}_train.jgz", train_payload)
        dump_jgz(output_dir / f"{category}_test.jgz", test_payload)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    total_train = sum(v["train_sequences"] for v in summary.values())
    total_test = sum(v["test_sequences"] for v in summary.values())
    total_missing = sum(v["missing"] for v in summary.values())
    print(f"Generated annotations under: {output_dir}")
    print(f"Train sequences: {total_train}")
    print(f"Test sequences: {total_test}")
    print(f"Missing frame entries skipped: {total_missing}")


if __name__ == "__main__":
    main()
