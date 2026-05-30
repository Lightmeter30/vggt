import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


DEFAULT_BLOCK_INDICES = (0, 4, 11, 17, 23)
DEFAULT_QUERY_FRAMES = ("first", "middle", "last")
DEFAULT_QUERY_KINDS = ("camera", "center_patch")


@dataclass
class AttentionCaptureConfig:
    output_dir: Path
    block_indices: Sequence[int] = field(default_factory=lambda: list(DEFAULT_BLOCK_INDICES))
    query_frames: Sequence[str] = field(default_factory=lambda: list(DEFAULT_QUERY_FRAMES))
    query_kinds: Sequence[str] = field(default_factory=lambda: list(DEFAULT_QUERY_KINDS))
    overlay_alpha: float = 0.5
    batch_index: int = 0

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.block_indices = tuple(int(index) for index in self.block_indices)
        self.query_frames = tuple(str(frame) for frame in self.query_frames)
        self.query_kinds = tuple(str(kind) for kind in self.query_kinds)
        self.overlay_alpha = float(self.overlay_alpha)
        if not 0.0 <= self.overlay_alpha <= 1.0:
            raise ValueError("overlay_alpha must be in [0, 1].")
        self.batch_index = int(self.batch_index)


@dataclass
class AttentionRecord:
    block_index: int
    query_name: str
    query_token_index: int
    key_attention: torch.Tensor
    patch_attention: torch.Tensor
    patch_grid: Tuple[int, int]
    sequence_length: int
    patch_start_idx: int
    attention_type: str = "global"


@dataclass
class AttentionCaptureContext:
    session: "AttentionCaptureSession"
    block_index: int
    attention_type: str
    batch_size: int
    sequence_length: int
    patch_start_idx: int
    patch_grid: Tuple[int, int]
    token_count: int

    def capture(self, q: torch.Tensor, k: torch.Tensor) -> None:
        self.session.capture_attention(q=q, k=k, context=self)


class AttentionCaptureSession:
    def __init__(self, config: AttentionCaptureConfig):
        self.config = config
        self.records: List[AttentionRecord] = []

    def make_context(
        self,
        *,
        block_index: int,
        attention_type: str,
        batch_size: int,
        sequence_length: int,
        patch_start_idx: int,
        patch_grid: Tuple[int, int],
        token_count: int,
    ) -> AttentionCaptureContext:
        return AttentionCaptureContext(
            session=self,
            block_index=int(block_index),
            attention_type=str(attention_type),
            batch_size=int(batch_size),
            sequence_length=int(sequence_length),
            patch_start_idx=int(patch_start_idx),
            patch_grid=(int(patch_grid[0]), int(patch_grid[1])),
            token_count=int(token_count),
        )

    def context_provider(self, **metadata) -> AttentionCaptureContext:
        return self.make_context(**metadata)

    def capture_attention(self, q: torch.Tensor, k: torch.Tensor, context: AttentionCaptureContext) -> None:
        if context.attention_type != "global":
            return
        if context.block_index not in self.config.block_indices:
            return

        patch_height, patch_width = context.patch_grid
        patch_token_count = patch_height * patch_width
        tokens_per_frame = context.patch_start_idx + patch_token_count
        expected_tokens = context.sequence_length * tokens_per_frame
        if context.token_count != expected_tokens:
            raise ValueError(
                f"Unexpected token count for attention capture: got {context.token_count}, "
                f"expected {expected_tokens}."
            )

        query_specs = self._resolve_query_specs(
            sequence_length=context.sequence_length,
            patch_start_idx=context.patch_start_idx,
            patch_grid=context.patch_grid,
        )
        if not query_specs:
            return

        batch_index = min(max(self.config.batch_index, 0), q.shape[0] - 1)
        query_indices = torch.tensor(
            [spec[2] for spec in query_specs],
            device=q.device,
            dtype=torch.long,
        )
        with torch.no_grad():
            q_selected = q[batch_index, :, query_indices, :]
            k_selected_batch = k[batch_index]
            logits = torch.einsum("hqd,hnd->hqn", q_selected, k_selected_batch)
            logits = logits * (q.shape[-1] ** -0.5)
            attention = logits.float().softmax(dim=-1).mean(dim=0).detach().cpu()

        for query_offset, (frame_index, query_kind, query_token_index) in enumerate(query_specs):
            key_attention = attention[query_offset]
            patch_attention = self._extract_patch_attention(
                key_attention=key_attention,
                sequence_length=context.sequence_length,
                patch_start_idx=context.patch_start_idx,
                patch_grid=context.patch_grid,
            )
            self.records.append(
                AttentionRecord(
                    block_index=context.block_index,
                    attention_type=context.attention_type,
                    query_name=f"frame_{frame_index:03d}_{query_kind}",
                    query_token_index=query_token_index,
                    key_attention=key_attention,
                    patch_attention=patch_attention,
                    patch_grid=context.patch_grid,
                    sequence_length=context.sequence_length,
                    patch_start_idx=context.patch_start_idx,
                )
            )

    def write_outputs(
        self,
        *,
        checkpoint_path: str,
        image_paths: Optional[Sequence[str]],
        input_images: torch.Tensor,
        preprocess_mode: str,
    ) -> Path:
        run_dir = _next_run_dir(self.config.output_dir)
        run_dir.mkdir(parents=True, exist_ok=False)

        images = _normalize_input_images(input_images)
        image_paths = list(image_paths or [])
        manifest_records = []

        for record in self.records:
            query_dir = run_dir / f"block_{record.block_index:03d}" / f"query_{_safe_name(record.query_name)}"
            query_dir.mkdir(parents=True, exist_ok=True)
            output_images = []
            for frame_index in range(record.sequence_length):
                frame_name = _frame_stem(image_paths, frame_index)
                output_path = query_dir / f"frame_{frame_index:03d}_{frame_name}.png"
                overlay = _build_overlay(
                    image=images[frame_index],
                    attention_map=record.patch_attention[frame_index],
                    alpha=self.config.overlay_alpha,
                )
                overlay.save(output_path)
                output_images.append(str(output_path.relative_to(run_dir)))

            manifest_records.append(
                {
                    "block_index": int(record.block_index),
                    "attention_type": record.attention_type,
                    "query_name": record.query_name,
                    "query_token_index": int(record.query_token_index),
                    "patch_grid": list(record.patch_grid),
                    "sequence_length": int(record.sequence_length),
                    "patch_start_idx": int(record.patch_start_idx),
                    "key_attention_sum": float(record.key_attention.sum().item()),
                    "output_images": output_images,
                }
            )

        manifest = {
            "checkpoint_path": str(checkpoint_path),
            "preprocess_mode": str(preprocess_mode),
            "image_paths": [str(path) for path in image_paths],
            "output_dir": str(run_dir),
            "records": manifest_records,
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return run_dir

    def _resolve_query_specs(
        self,
        *,
        sequence_length: int,
        patch_start_idx: int,
        patch_grid: Tuple[int, int],
    ) -> List[Tuple[int, str, int]]:
        frame_indices = _resolve_frame_indices(self.config.query_frames, sequence_length)
        patch_height, patch_width = patch_grid
        center_patch_index = (patch_height // 2) * patch_width + (patch_width // 2)
        patch_token_count = patch_height * patch_width
        tokens_per_frame = patch_start_idx + patch_token_count

        specs: List[Tuple[int, str, int]] = []
        seen = set()
        for frame_index in frame_indices:
            frame_start = frame_index * tokens_per_frame
            for query_kind in self.config.query_kinds:
                if query_kind == "camera":
                    token_index = frame_start
                elif query_kind == "center_patch":
                    token_index = frame_start + patch_start_idx + center_patch_index
                else:
                    raise ValueError(f"Unsupported attention query kind: {query_kind}")
                key = (frame_index, query_kind, token_index)
                if key not in seen:
                    specs.append(key)
                    seen.add(key)
        return specs

    @staticmethod
    def _extract_patch_attention(
        *,
        key_attention: torch.Tensor,
        sequence_length: int,
        patch_start_idx: int,
        patch_grid: Tuple[int, int],
    ) -> torch.Tensor:
        patch_height, patch_width = patch_grid
        patch_token_count = patch_height * patch_width
        tokens_per_frame = patch_start_idx + patch_token_count
        patch_maps = []
        for frame_index in range(sequence_length):
            frame_start = frame_index * tokens_per_frame
            patch_start = frame_start + patch_start_idx
            patch_end = patch_start + patch_token_count
            patch_maps.append(key_attention[patch_start:patch_end].reshape(patch_height, patch_width))
        return torch.stack(patch_maps, dim=0)


def _resolve_frame_indices(frame_specs: Sequence[str], sequence_length: int) -> List[int]:
    resolved: List[int] = []
    seen = set()
    for frame_spec in frame_specs:
        spec = str(frame_spec)
        if spec == "first":
            frame_index = 0
        elif spec == "middle":
            frame_index = sequence_length // 2
        elif spec == "last":
            frame_index = sequence_length - 1
        else:
            frame_index = int(spec)
        if frame_index < 0:
            frame_index = sequence_length + frame_index
        if not 0 <= frame_index < sequence_length:
            raise ValueError(f"Query frame {frame_spec} is outside sequence length {sequence_length}.")
        if frame_index not in seen:
            resolved.append(frame_index)
            seen.add(frame_index)
    return resolved


def _normalize_input_images(input_images: torch.Tensor) -> torch.Tensor:
    images = input_images.detach().float().cpu()
    if images.ndim == 5:
        images = images[0]
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("input_images must have shape [S, 3, H, W] or [B, S, 3, H, W].")
    return images.clamp(0.0, 1.0)


def _build_overlay(image: torch.Tensor, attention_map: torch.Tensor, alpha: float) -> Image.Image:
    _, height, width = image.shape
    heatmap = F.interpolate(
        attention_map.detach().float().view(1, 1, *attention_map.shape),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    heatmap = _normalize_map(heatmap).numpy()
    heatmap_rgb = _colorize_heatmap(heatmap)
    image_rgb = (image.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    overlay = ((1.0 - alpha) * image_rgb.astype(np.float32) + alpha * heatmap_rgb.astype(np.float32)).clip(0, 255)
    return Image.fromarray(overlay.astype(np.uint8))


def _normalize_map(values: torch.Tensor) -> torch.Tensor:
    if torch.isnan(values).any():
        return torch.zeros_like(values)
    min_value = values.min()
    max_value = values.max()
    denom = max_value - min_value
    # 注意：float32 的 epsilon 约为 1.19e-7，这里用 1e-12 对 float64
    # 是安全的，但对 float32 输入可能防护不足。使用 abs(denom) 同时
    # 捕获浮点比较中 NaN 的 IEEE 754 行为（NaN < x 始终为 False）。
    if abs(float(denom)) < 1e-12:
        return torch.zeros_like(values)
    return (values - min_value) / denom


def _colorize_heatmap(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(np.float32), 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    return (np.stack([red, green, blue], axis=-1) * 255.0).astype(np.uint8)


def _next_run_dir(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    for index in range(10000):
        candidate = output_dir / f"run_{index:03d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate unique attention output directory under {output_dir}.")


def _frame_stem(image_paths: Sequence[str], frame_index: int) -> str:
    if frame_index < len(image_paths):
        stem = Path(image_paths[frame_index]).stem
    else:
        stem = f"image_{frame_index:03d}"
    return _safe_name(stem)


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return safe or "item"


def parse_int_list(value: str) -> List[int]:
    return [int(item) for item in _split_csv(value)]


def parse_str_list(value: str) -> List[str]:
    return _split_csv(value)


def _split_csv(value: str) -> List[str]:
    parts: List[str] = []
    for chunk in str(value).split(","):
        item = chunk.strip()
        if item:
            parts.append(item)
    if not parts:
        raise ValueError("Expected at least one comma-separated value.")
    return parts
