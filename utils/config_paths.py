"""Resolve paths in config relative to the config file directory (portable, no hard-coded drives)."""

from pathlib import Path


def resolve_path_value(value: str | Path, config_dir: Path) -> Path:
    """If path is relative, resolve against the directory containing config.yaml."""
    path = Path(value)
    if path.is_absolute():
        return path.expanduser().resolve()
    return (config_dir / path).expanduser().resolve()
