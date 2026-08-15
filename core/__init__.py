import json
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
OUTPUT_DIR = ROOT_DIR / "output"
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "default.json"


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(path) as f:
        return json.load(f)
