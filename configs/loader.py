import yaml
from pathlib import Path

_DIR = Path(__file__).parent


def load_config(name: str) -> dict:
    """Load a YAML config file relative to the configs/ directory.

    Supports one `extends: <path>` key (resolved relative to the file's own
    directory, .yaml suffix optional): the parent loads first, then this
    file's keys merge on top of it -- a dict-valued key (training:/qat:/data:)
    merges field-by-field instead of replacing the whole block, so a child can
    override just one hyperparameter. Used by configs/experiments/*.yaml to
    share a protocol (configs/experiments/_protocols/*.yaml) across several
    experiments without repeating it in each file.
    """
    path = _DIR / name
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    parent_name = cfg.pop("extends", None)
    if parent_name is None:
        return cfg

    parent_path = path.parent / parent_name
    if parent_path.suffix != ".yaml":
        parent_path = parent_path.with_suffix(".yaml")
    parent_cfg = load_config(str(parent_path.relative_to(_DIR)))

    merged = dict(parent_cfg)
    for key, value in cfg.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged
