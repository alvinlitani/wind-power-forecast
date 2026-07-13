"""
Storage abstraction over local disk and Google Cloud Storage.

Every data-touching script reads/writes through this module instead of
calling pandas / open() / Path directly. Whether a path resolves to the
local filesystem or to a GCS bucket is decided purely by the path string:

    - "data/predictions/pc/x.csv"               -> local disk
    - "gs://wind-power-ontario-data/.../x.csv"  -> GCS

This lets the exact same script run on a laptop (DATA_ROOT=data) and inside
the Prefect worker (DATA_ROOT=gs://wind-power-ontario-data) with no code
change — only the DATA_ROOT environment variable differs.

Configuration
-------------
Storage roots come from `wind_forecast.config` (DATA_ROOT, MODELS_ROOT),
each set via environment variable. Set them to gs:// URIs in the cloud,
leave at local defaults ("data" / "models") for local runs.

Public API
----------
    data_path(*parts)            -> join parts onto DATA_ROOT
    models_path(*parts)          -> join parts onto MODELS_ROOT
    read_csv(path, **kwargs)     -> pd.DataFrame
    write_csv(df, path, **kwargs)-> None
    read_text(path)              -> str
    read_bytes(path)             -> bytes
    write_bytes(data, path)      -> None
    exists(path)                 -> bool
    list_files(prefix)           -> list[str]
    glob(pattern)                -> list[str]
    open_file(path, mode)        -> file-like context manager
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

from wind_forecast import config


def _is_gcs(path: str) -> bool:
    return str(path).startswith("gs://")


# gcsfs filesystem is created lazily and reused. Importing/constructing it is
# only needed when a gs:// path is actually used, so local-only runs never
# require the dependency or credentials.
_gcs_fs = None


def _fs():
    """Return a cached gcsfs filesystem, importing the dependency on demand."""
    global _gcs_fs
    if _gcs_fs is None:
        import gcsfs  # imported lazily so local runs don't need it installed

        _gcs_fs = gcsfs.GCSFileSystem()
    return _gcs_fs


def data_path(*parts: str) -> str:
    """Join path parts onto DATA_ROOT.

    Uses forward slashes throughout (valid for both local POSIX paths and
    gs:// URIs). Example:
        data_path("predictions", "pc", "x.csv")
        -> "data/predictions/pc/x.csv"          (local)
        -> "gs://bucket/predictions/pc/x.csv"    (cloud)
    """
    root = config.DATA_ROOT.rstrip("/")
    suffix = "/".join(str(p).strip("/") for p in parts)
    return f"{root}/{suffix}" if suffix else root


def models_path(*parts: str) -> str:
    """Join path parts onto MODELS_ROOT (separate bucket from data)."""
    root = config.MODELS_ROOT.rstrip("/")
    suffix = "/".join(str(p).strip("/") for p in parts)
    return f"{root}/{suffix}" if suffix else root


def exists(path: str) -> bool:
    """True if the file exists, on whichever backend the path points to."""
    if _is_gcs(path):
        return _fs().exists(path)
    return Path(path).exists()


def _ensure_local_parent(path: str) -> None:
    """Create parent directories for a local path. No-op for GCS (no dirs)."""
    if not _is_gcs(path):
        parent = Path(path).parent
        if str(parent):
            parent.mkdir(parents=True, exist_ok=True)


def read_csv(path: str, **kwargs) -> pd.DataFrame:
    """Read a CSV into a DataFrame from local disk or GCS.

    Extra kwargs pass through to pandas.read_csv (parse_dates, dtype, etc).
    pandas reads gs:// URIs directly when gcsfs is installed.
    """
    return pd.read_csv(path, **kwargs)


def write_csv(df: pd.DataFrame, path: str, index: bool = False, **kwargs) -> None:
    """Write a DataFrame to CSV on local disk or GCS.

    Creates local parent directories automatically (mirrors the old
    Path.mkdir(parents=True) calls). Defaults index=False since none of the
    pipeline outputs want the pandas index column.
    """
    _ensure_local_parent(path)
    df.to_csv(path, index=index, **kwargs)


def read_text(path: str, encoding: str = "utf-8") -> str:
    """Return the full file contents as a string."""
    if _is_gcs(path):
        with _fs().open(path, "r") as f:
            return f.read()
    return Path(path).read_text(encoding=encoding)


def read_bytes(path: str) -> bytes:
    """Return the full file contents as bytes (for pickle / torch artifacts)."""
    if _is_gcs(path):
        with _fs().open(path, "rb") as f:
            return f.read()
    return Path(path).read_bytes()


def write_bytes(data: bytes, path: str) -> None:
    """Write raw bytes to local disk or GCS (for pickle / torch artifacts)."""
    _ensure_local_parent(path)
    if _is_gcs(path):
        with _fs().open(path, "wb") as f:
            f.write(data)
    else:
        Path(path).write_bytes(data)


@contextmanager
def open_file(path: str, mode: str = "r", encoding: str | None = "utf-8") -> Iterator:
    """Open a file handle on either backend, usable as a context manager.

    Lets code that wants a real file object (e.g. csv.DictReader, pickle.load)
    work unchanged across local and GCS:

        with storage.open_file(path, "r") as f:
            reader = csv.DictReader(f)

    Binary modes ("rb"/"wb") ignore the encoding argument.
    """
    binary = "b" in mode
    if "w" in mode or "a" in mode:
        _ensure_local_parent(path)

    if _is_gcs(path):
        f = _fs().open(path, mode) if binary else _fs().open(path, mode, encoding=encoding)
        try:
            yield f
        finally:
            f.close()
    else:
        f = open(path, mode) if binary else open(path, mode, encoding=encoding)
        try:
            yield f
        finally:
            f.close()


def list_files(prefix: str) -> list[str]:
    """List files directly under a prefix/directory (non-recursive).

    Returns full paths. For GCS the returned entries are gs:// URIs.
    """
    if _is_gcs(prefix):
        fs = _fs()
        if not fs.exists(prefix):
            return []
        return [f"gs://{p}" for p in fs.ls(prefix) if not p.endswith("/")]
    p = Path(prefix)
    if not p.exists():
        return []
    return [child.as_posix() for child in p.iterdir() if child.is_file()]


def glob(pattern: str) -> list[str]:
    """Glob for files matching a wildcard pattern (e.g. '.../pc/*.csv').

    Returns full paths; GCS entries come back as gs:// URIs.
    """
    if _is_gcs(pattern):
        return [f"gs://{p}" for p in _fs().glob(pattern)]
    # Local: split the fixed root from the wildcard tail so Path.glob works.
    # Find the first path segment containing a wildcard char.
    parts = Path(pattern).parts
    for i, part in enumerate(parts):
        if any(c in part for c in "*?[]"):
            root = Path(*parts[:i]) if i > 0 else Path(".")
            tail = str(Path(*parts[i:]))
            return [m.as_posix() for m in root.glob(tail)]
    # No wildcard at all: return the path if it exists.
    return [pattern] if Path(pattern).exists() else []
