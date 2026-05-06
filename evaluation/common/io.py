import gzip
import json
from pathlib import Path


def load_json_gz(path):
    with gzip.open(Path(path), "rt", encoding="utf-8") as fin:
        return json.load(fin)
