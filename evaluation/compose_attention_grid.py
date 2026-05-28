import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vggt.utils.attention_visualization import parse_int_list  # noqa: E402


DEFAULT_CELL_WIDTH = 300
DEFAULT_GAP = 16
DEFAULT_LEFT_MARGIN = 92
DEFAULT_TOP_MARGIN = 64
DEFAULT_RIGHT_MARGIN = 24
DEFAULT_BOTTOM_MARGIN = 24


def build_attention_grid(
    *,
    run_dir: Path,
    output_path: Optional[Path] = None,
    frame_index: int = 0,
    blocks: Optional[Sequence[int]] = None,
    tokens: Optional[Sequence[int]] = None,
    cell_width: int = DEFAULT_CELL_WIDTH,
    cell_size: Optional[Tuple[int, int]] = None,
    gap: int = DEFAULT_GAP,
    label_mode: str = "token",
) -> Path:
    run_dir = Path(run_dir)
    manifest = _load_manifest(run_dir)
    records = list(manifest.get("records", []))
    if not records:
        raise ValueError(f"No attention records found in {run_dir / 'manifest.json'}.")

    selected_blocks = list(blocks) if blocks is not None else _unique(record["block_index"] for record in records)
    selected_tokens = list(tokens) if tokens is not None else _unique(record["query_token_index"] for record in records)
    record_map = _build_record_map(records)

    first_image_path = _resolve_record_image(run_dir, records[0], frame_index)
    first_size = Image.open(first_image_path).size
    if cell_size is None:
        cell_width = int(cell_width)
        cell_height = max(1, round(cell_width * first_size[1] / first_size[0]))
    else:
        cell_width, cell_height = int(cell_size[0]), int(cell_size[1])

    output_path = Path(output_path) if output_path is not None else run_dir / f"attention_grid_frame_{frame_index:03d}.png"
    image = _render_grid(
        run_dir=run_dir,
        record_map=record_map,
        blocks=selected_blocks,
        tokens=selected_tokens,
        frame_index=frame_index,
        cell_size=(cell_width, cell_height),
        gap=int(gap),
        label_mode=label_mode,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compose a block-by-token attention grid from evaluation/visualize_attention.py outputs."
    )
    parser.add_argument("--run_dir", type=Path, required=True, help="Attention run directory containing manifest.json.")
    parser.add_argument("--output_path", type=Path, default=None, help="Output PNG path.")
    parser.add_argument("--frame_index", type=int, default=0, help="Which per-record output frame to use in each cell.")
    parser.add_argument("--blocks", type=parse_int_list, default=None, help="Comma-separated block indices to show.")
    parser.add_argument("--tokens", type=parse_int_list, default=None, help="Comma-separated query token indices to show.")
    parser.add_argument("--cell_width", type=int, default=DEFAULT_CELL_WIDTH, help="Output cell width in pixels.")
    parser.add_argument(
        "--cell_height",
        type=int,
        default=None,
        help="Output cell height in pixels. Defaults to preserving source aspect ratio from --cell_width.",
    )
    parser.add_argument("--gap", type=int, default=DEFAULT_GAP, help="Gap between grid cells in pixels.")
    parser.add_argument(
        "--label_mode",
        choices=("token", "query", "both"),
        default="token",
        help="Column label style.",
    )
    return parser


def main(argv=None) -> Path:
    args = build_parser().parse_args(argv)
    cell_size = None
    if args.cell_height is not None:
        cell_size = (args.cell_width, args.cell_height)
    output_path = build_attention_grid(
        run_dir=args.run_dir,
        output_path=args.output_path,
        frame_index=args.frame_index,
        blocks=args.blocks,
        tokens=args.tokens,
        cell_width=args.cell_width,
        cell_size=cell_size,
        gap=args.gap,
        label_mode=args.label_mode,
    )
    print(f"Saved attention grid to: {output_path}")
    return output_path


def _render_grid(
    *,
    run_dir: Path,
    record_map: Dict[Tuple[int, int], dict],
    blocks: Sequence[int],
    tokens: Sequence[int],
    frame_index: int,
    cell_size: Tuple[int, int],
    gap: int,
    label_mode: str,
) -> Image.Image:
    cell_width, cell_height = cell_size
    width = DEFAULT_LEFT_MARGIN + len(tokens) * cell_width + max(0, len(tokens) - 1) * gap + DEFAULT_RIGHT_MARGIN
    height = DEFAULT_TOP_MARGIN + len(blocks) * cell_height + max(0, len(blocks) - 1) * gap + DEFAULT_BOTTOM_MARGIN
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    column_font = _load_font(36)
    row_font = _load_font(34)

    for column_index, token in enumerate(tokens):
        x = DEFAULT_LEFT_MARGIN + column_index * (cell_width + gap)
        record = _find_first_record_for_token(record_map, token)
        label = _column_label(record, label_mode)
        _draw_centered_text(draw, label, column_font, x, 10, cell_width)

    for row_index, block in enumerate(blocks):
        y = DEFAULT_TOP_MARGIN + row_index * (cell_height + gap)
        row_label = _rotated_text(f"Block {block}", row_font)
        label_x = max(0, (DEFAULT_LEFT_MARGIN - row_label.width) // 2)
        label_y = y + max(0, (cell_height - row_label.height) // 2)
        canvas.paste(row_label, (label_x, label_y), row_label)

    for row_index, block in enumerate(blocks):
        for column_index, token in enumerate(tokens):
            record = _require_record(record_map, block, token)
            source_path = _resolve_record_image(run_dir, record, frame_index)
            cell = Image.open(source_path).convert("RGB").resize(cell_size, Image.Resampling.BICUBIC)
            x = DEFAULT_LEFT_MARGIN + column_index * (cell_width + gap)
            y = DEFAULT_TOP_MARGIN + row_index * (cell_height + gap)
            canvas.paste(cell, (x, y))

    return canvas


def _load_manifest(run_dir: Path) -> dict:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing attention manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _build_record_map(records: Sequence[dict]) -> Dict[Tuple[int, int], dict]:
    record_map: Dict[Tuple[int, int], dict] = {}
    for record in records:
        key = (int(record["block_index"]), int(record["query_token_index"]))
        record_map.setdefault(key, record)
    return record_map


def _resolve_record_image(run_dir: Path, record: dict, frame_index: int) -> Path:
    output_images = list(record.get("output_images", []))
    if not 0 <= frame_index < len(output_images):
        raise ValueError(
            f"frame_index={frame_index} is outside output_images for "
            f"block={record.get('block_index')} token={record.get('query_token_index')}."
        )
    image_path = run_dir / output_images[frame_index]
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing attention cell image: {image_path}")
    return image_path


def _require_record(record_map: Dict[Tuple[int, int], dict], block: int, token: int) -> dict:
    key = (int(block), int(token))
    if key not in record_map:
        raise ValueError(f"No attention record for block={block}, token={token}.")
    return record_map[key]


def _find_first_record_for_token(record_map: Dict[Tuple[int, int], dict], token: int) -> dict:
    for (_block, record_token), record in record_map.items():
        if record_token == int(token):
            return record
    raise ValueError(f"No attention record for token={token}.")


def _column_label(record: dict, label_mode: str) -> str:
    token_label = f"Token {int(record['query_token_index'])}"
    if label_mode == "token":
        return token_label
    query_label = str(record.get("query_name", token_label))
    if label_mode == "query":
        return query_label
    return f"{token_label}\n{query_label}"


def _draw_centered_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, x: int, y: int, width: int) -> None:
    lines = text.splitlines()
    line_heights = []
    line_widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])
    total_height = sum(line_heights) + max(0, len(lines) - 1) * 2
    cursor_y = y + max(0, (DEFAULT_TOP_MARGIN - total_height) // 2) - 4
    for line, line_width, line_height in zip(lines, line_widths, line_heights):
        draw.text((x + (width - line_width) // 2, cursor_y), line, fill=(0, 0, 0), font=font)
        cursor_y += line_height + 2


def _rotated_text(text: str, font: ImageFont.ImageFont) -> Image.Image:
    scratch = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
    draw = ImageDraw.Draw(scratch)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_image = Image.new("RGBA", (bbox[2] - bbox[0] + 8, bbox[3] - bbox[1] + 8), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_image)
    text_draw.text((4 - bbox[0], 4 - bbox[1]), text, fill=(0, 0, 0, 255), font=font)
    return text_image.rotate(90, expand=True)


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _unique(values: Iterable[int]) -> List[int]:
    result: List[int] = []
    seen = set()
    for value in values:
        value = int(value)
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


if __name__ == "__main__":
    main()
