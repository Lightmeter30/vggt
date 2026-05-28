import json
from pathlib import Path

from PIL import Image

from evaluation.compose_attention_grid import build_attention_grid


def _write_cell(run_dir: Path, rel_path: str, color: tuple[int, int, int]) -> str:
    path = run_dir / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 8), color=color).save(path)
    return rel_path


def test_build_attention_grid_uses_manifest_block_and_token_layout(tmp_path):
    run_dir = tmp_path / "run_000"
    records = []
    for block_index in (0, 4):
        for token_index, color in ((10, (255, 0, 0)), (20, (0, 255, 0))):
            rel_path = _write_cell(
                run_dir,
                f"block_{block_index:03d}/query_token_{token_index}/frame_000.png",
                color,
            )
            records.append(
                {
                    "block_index": block_index,
                    "query_name": f"token_{token_index}",
                    "query_token_index": token_index,
                    "output_images": [rel_path],
                }
            )
    manifest = {"records": records}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    output_path = build_attention_grid(
        run_dir=run_dir,
        output_path=tmp_path / "grid.png",
        frame_index=0,
        blocks=[4, 0],
        tokens=[20, 10],
        cell_size=(24, 16),
    )

    assert output_path.is_file()
    image = Image.open(output_path)
    assert image.size[0] > 48
    assert image.size[1] > 32
