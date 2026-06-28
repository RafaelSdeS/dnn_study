import yaml
from pathlib import Path

_DIR = Path(__file__).parent


def load_config(name: str) -> dict:
    """Load a YAML config file relative to the configs/ directory."""
    with open(_DIR / name) as f:
        return yaml.safe_load(f) or {}
