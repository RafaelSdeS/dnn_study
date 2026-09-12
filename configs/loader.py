import yaml
from pathlib import Path

_DIR = Path(__file__).parent


def _resolve_extends(node, base_dir: Path):
    """Recursively resolve `extends:` keys, at the top level or nested inside any block."""
    if not isinstance(node, dict):
        return node

    resolved = {k: _resolve_extends(v, base_dir) for k, v in node.items()}
    parent_name = resolved.pop("extends", None)
    if parent_name is None:
        return resolved

    parent_path = base_dir / parent_name
    if parent_path.suffix != ".yaml":
        parent_path = parent_path.with_suffix(".yaml")
    with open(parent_path) as f:
        parent_cfg = _resolve_extends(yaml.safe_load(f) or {}, parent_path.parent)

    merged = dict(parent_cfg)
    for key, value in resolved.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def load_config(name: str) -> dict:
    """Load a YAML config file relative to the configs/ directory.

    Supports `extends: <path>` keys (resolved relative to the extending block's
    own file, .yaml suffix optional), at the top level of the file or nested
    inside any block: the parent loads first, then the block's keys merge on
    top of it -- a dict-valued key (training:/qat:/data:) merges field-by-field
    instead of replacing the whole block, so a child can override just one
    hyperparameter. Used by configs/experiments/*.yaml to share a protocol
    (configs/experiments/_protocols/*.yaml) across several experiments without
    repeating it in each file, and by configs/compression.yaml to reuse
    configs/qat.yaml's recipe per block instead of copying its values.
    """
    path = _DIR / name
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return _resolve_extends(cfg, path.parent)
