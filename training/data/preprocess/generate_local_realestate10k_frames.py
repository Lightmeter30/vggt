import argparse
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2


MICROSECONDS_PER_SECOND = 1_000_000.0


def frame_tolerance_us(fps):
    if fps <= 0:
        raise ValueError(f"FPS must be positive, got {fps}.")
    return MICROSECONDS_PER_SECOND / (2.0 * fps)


def extract_youtube_id(url):
    parsed = urlparse(url.strip())
    query_video_ids = parse_qs(parsed.query).get("v")
    if query_video_ids:
        return query_video_ids[0]

    path_parts = [part for part in parsed.path.split("/") if part]
    if path_parts:
        return path_parts[-1]
    return None


def parse_frame_line(line):
    values = line.strip().split()
    if len(values) != 19:
        return None

    return {
        "timestamp": values[0],
        "timestamp_us": int(values[0]),
        "raw_line": line.rstrip("\n"),
    }


def load_metadata_sequence(metadata_path):
    lines = Path(metadata_path).read_text(encoding="utf-8").splitlines()
    if not lines:
        return None

    video_id = extract_youtube_id(lines[0])
    if not video_id:
        return None

    frames = []
    for line in lines[1:]:
        frame = parse_frame_line(line)
        if frame is not None:
            frames.append(frame)

    return {
        "seq_name": Path(metadata_path).stem,
        "metadata_path": str(metadata_path),
        "video_url": lines[0].strip(),
        "video_id": video_id,
        "frames": frames,
    }


def resolve_downloaded_video_path(realestate10k_dir, video_id):
    downloaded_root = Path(realestate10k_dir) / "downloaded"
    target_path = downloaded_root / video_id
    if target_path.is_file():
        return target_path

    matches = sorted(downloaded_root.glob(f"{video_id}.*"))
    for match in matches:
        if match.is_file():
            return match
    return None


def find_best_metadata_match(video_time_us, metadata_frames, tolerance_us, used_indices):
    best_index = None
    best_distance = None

    for index, metadata_frame in enumerate(metadata_frames):
        if index in used_indices:
            continue

        metadata_time_us = int(metadata_frame.get("timestamp_us", metadata_frame["timestamp"]))
        distance = abs(metadata_time_us - int(round(video_time_us)))
        if distance > tolerance_us:
            continue

        if best_distance is None or distance < best_distance:
            best_index = index
            best_distance = distance

    if best_index is None:
        return None, None
    return metadata_frames[best_index], int(best_distance)


def _write_image(image_output_path, image, overwrite):
    image_output_path = Path(image_output_path)
    image_output_path.parent.mkdir(parents=True, exist_ok=True)
    if image_output_path.exists() and not overwrite:
        return
    if not cv2.imwrite(str(image_output_path), image):
        raise RuntimeError(f"Failed to write image: {image_output_path}")


def extract_sequence_frames(sequence, realestate10k_dir, output_dir, overwrite=False):
    video_path = resolve_downloaded_video_path(realestate10k_dir, sequence["video_id"])
    if video_path is None:
        return [], {"missing_video": 1, "bad_video": 0, "matched_frames": 0}

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        return [], {"missing_video": 0, "bad_video": 1, "matched_frames": 0}

    fps = float(video.get(cv2.CAP_PROP_FPS))
    try:
        tolerance_us = frame_tolerance_us(fps)
    except ValueError:
        video.release()
        return [], {"missing_video": 0, "bad_video": 1, "matched_frames": 0}
    used_indices = set()
    manifest_rows = []

    while video.isOpened():
        frame_ok, image = video.read()
        if not frame_ok:
            break

        video_time_us = float(video.get(cv2.CAP_PROP_POS_MSEC)) * 1000.0
        metadata_frame, abs_error_us = find_best_metadata_match(
            video_time_us=video_time_us,
            metadata_frames=sequence["frames"],
            tolerance_us=tolerance_us,
            used_indices=used_indices,
        )
        if metadata_frame is None:
            continue

        matched_index = next(
            index for index, candidate in enumerate(sequence["frames"]) if candidate is metadata_frame
        )
        used_indices.add(matched_index)

        image_output_path = Path(output_dir) / sequence["video_id"] / f"{metadata_frame['timestamp']}.jpg"
        _write_image(image_output_path, image, overwrite=overwrite)
        manifest_rows.append(
            {
                "seq_name": sequence["seq_name"],
                "video_id": sequence["video_id"],
                "timestamp": metadata_frame["timestamp"],
                "matched_video_time_us": int(round(video_time_us)),
                "abs_error_us": float(abs_error_us),
                "fps": fps,
                "image_path": str(image_output_path),
                "source_video_path": str(video_path),
            }
        )

        if len(used_indices) == len(sequence["frames"]):
            break

    video.release()
    return manifest_rows, {
        "missing_video": 0,
        "bad_video": 0,
        "matched_frames": len(manifest_rows),
    }


def iter_metadata_paths(realestate10k_dir, split):
    split_dir = Path(realestate10k_dir) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"RealEstate10K split directory not found: {split_dir}")
    return sorted(split_dir.glob("*.txt"))


def generate_frames(
    realestate10k_dir,
    split,
    output_dir=None,
    manifest_path=None,
    overwrite=False,
    limit_sequences=None,
):
    realestate10k_dir = Path(realestate10k_dir)
    output_dir = Path(output_dir) if output_dir is not None else realestate10k_dir / "transcode"
    manifest_path = Path(manifest_path) if manifest_path is not None else realestate10k_dir / "transcode_manifest.jsonl"
    metadata_paths = iter_metadata_paths(realestate10k_dir, split)
    if limit_sequences is not None:
        metadata_paths = metadata_paths[:limit_sequences]

    metadata_paths = list(metadata_paths)
    total = len(metadata_paths)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "sequences_total": 0,
        "sequences_with_matches": 0,
        "missing_video": 0,
        "bad_video": 0,
        "matched_frames": 0,
    }

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for idx, metadata_path in enumerate(metadata_paths, start=1):
            sequence = load_metadata_sequence(metadata_path)
            if sequence is None:
                print(f"[{idx}/{total}] {metadata_path.stem} 跳过（无效元数据）")
                continue

            summary["sequences_total"] += 1
            rows, stats = extract_sequence_frames(
                sequence=sequence,
                realestate10k_dir=realestate10k_dir,
                output_dir=output_dir,
                overwrite=overwrite,
            )
            if rows:
                summary["sequences_with_matches"] += 1
            summary["missing_video"] += stats["missing_video"]
            summary["bad_video"] += stats["bad_video"]
            summary["matched_frames"] += stats["matched_frames"]

            for row in rows:
                manifest_file.write(json.dumps(row, sort_keys=True) + "\n")

            status = f"+{stats['matched_frames']}帧" if rows else ("无视频" if stats["missing_video"] else "视频异常")
            print(
                f"[{idx}/{total}] {sequence['seq_name']}  {status}  "
                f"累计匹配: {summary['matched_frames']}帧/{summary['sequences_with_matches']}序列"
            )

    return summary, manifest_path


def build_parser():
    parser = argparse.ArgumentParser(description="Extract RealEstate10K frames with microsecond timestamp matching.")
    parser.add_argument("--realestate10k_dir", required=True, help="Path to the RealEstate10K dataset root.")
    parser.add_argument("--split", default="test", choices=("train", "test"), help="Metadata split to process.")
    parser.add_argument("--output_dir", default=None, help="Directory for extracted transcode/<video_id> frames.")
    parser.add_argument("--manifest_path", default=None, help="JSONL manifest path.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing extracted JPG frames.")
    parser.add_argument("--limit_sequences", type=int, default=None, help="Only process the first N metadata files.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary, manifest_path = generate_frames(
        realestate10k_dir=args.realestate10k_dir,
        split=args.split,
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        overwrite=args.overwrite,
        limit_sequences=args.limit_sequences,
    )

    print(f"Manifest written to: {manifest_path}")
    for key in sorted(summary):
        print(f"{key}: {summary[key]}")
    return summary


if __name__ == "__main__":
    main()
