MODEL_REGISTRY: dict = {}


def register_model(name: str, ctor, fuse_map=None, **metadata) -> None:
    """Register an architecture (constructor + QAT fuse map + metadata) by name."""
    MODEL_REGISTRY[name] = {
        "ctor": ctor,
        "fuse_map": fuse_map or [],
        **metadata,
    }
