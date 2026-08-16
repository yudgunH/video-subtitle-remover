"""Portable application data paths.

The desktop build is intentionally an ``onedir`` application.  A tiny locator
next to the executable remembers the selected data directory; every mutable
file owned by VSR then lives below that directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


DATA_DIR_ENV = "VSR_DATA_DIR"
APP_ROOT_ENV = "VSR_APP_ROOT"
_LOCATION_FILE = ".vsr-data-location.json"


def application_root() -> Path:
    override = os.getenv(APP_ROOT_ENV, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def resource_root() -> Path:
    """Return the read-only source/bundle root containing models and assets."""

    bundle_root = getattr(sys, "_MEIPASS", None)
    return Path(bundle_root).resolve() if bundle_root else application_root()


def data_location_file() -> Path:
    return application_root() / _LOCATION_FILE


def _read_selected_data_root() -> Path | None:
    try:
        payload = json.loads(data_location_file().read_text(encoding="utf-8"))
        selected = str(payload.get("data_root", "")).strip()
        return Path(selected).expanduser().resolve() if selected else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def get_data_root() -> Path:
    selected = os.getenv(DATA_DIR_ENV, "").strip()
    if selected:
        return Path(selected).expanduser().resolve()
    return _read_selected_data_root() or (application_root() / "data")


def get_data_path(*parts: str, create_parent: bool = False) -> Path:
    path = get_data_root().joinpath(*parts)
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_data_directory(*parts: str) -> Path:
    path = get_data_root().joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def migrate_legacy_project_data() -> None:
    """Copy data created by pre-portable releases on the first new launch."""

    root = get_data_root()
    legacy_config = application_root() / "config" / "config.json"
    new_config = root / "config" / "config.json"
    if legacy_config != new_config and legacy_config.is_file() and not new_config.exists():
        new_config.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy_config, new_config)

    legacy_cache = application_root() / "backend" / "config" / "translation_cache.json"
    new_cache = root / "cache" / "translation_cache.json"
    if legacy_cache != new_cache and legacy_cache.is_file() and not new_cache.exists():
        new_cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy_cache, new_cache)


def _verify_writable(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / f".vsr-write-test-{os.getpid()}"
    try:
        probe.write_bytes(b"ok")
        probe.unlink()
    except OSError as error:
        raise OSError(f"Application data folder is not writable: {directory}") from error


def set_data_root(directory: str | os.PathLike, migrate: bool = True) -> Path:
    """Persist a new data root and optionally copy durable data into it.

    A restart is required after this call because qconfig keeps its current
    config file open for the lifetime of the process.
    """

    new_root = Path(directory).expanduser().resolve()
    old_root = get_data_root()
    _verify_writable(new_root)

    if migrate and old_root != new_root and old_root.exists():
        for name in ("config", "cache", "ocr_checkpoints", "secrets"):
            source = old_root / name
            target = new_root / name
            if source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True)

    pointer = data_location_file()
    pointer.parent.mkdir(parents=True, exist_ok=True)
    temporary = pointer.with_suffix(pointer.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"data_root": str(new_root)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, pointer)
    os.environ[DATA_DIR_ENV] = str(new_root)
    initialize_runtime_environment()
    return new_root


def initialize_runtime_environment() -> Path:
    """Route process caches and temporary files into the portable data root."""

    root = get_data_root()
    _verify_writable(root)
    temp_dir = get_data_directory("temp")
    cache_dir = get_data_directory("cache")

    paths = {
        "TMP": temp_dir,
        "TEMP": temp_dir,
        "TMPDIR": temp_dir,
        "XDG_CACHE_HOME": cache_dir,
        "TORCH_HOME": cache_dir / "torch",
        "TORCH_EXTENSIONS_DIR": cache_dir / "torch_extensions",
        "PADDLE_HOME": cache_dir / "paddle",
        "PADDLEOCR_HOME": cache_dir / "paddleocr",
        "PADDLE_PDX_CACHE_HOME": cache_dir / "paddlex",
        "HF_HOME": cache_dir / "huggingface",
        "HUGGINGFACE_HUB_CACHE": cache_dir / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": cache_dir / "huggingface" / "transformers",
        "MPLCONFIGDIR": cache_dir / "matplotlib",
        "CUDA_CACHE_PATH": cache_dir / "cuda",
        "NUMBA_CACHE_DIR": cache_dir / "numba",
    }
    for variable, value in paths.items():
        value.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(value)

    # tempfile caches the result of gettempdir(), so update it explicitly when
    # the user changes the data folder during a running GUI session.
    tempfile.tempdir = str(temp_dir)
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    return root
