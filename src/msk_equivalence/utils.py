from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


LOGGER_NAME = "msk_equivalence"


def setup_logging(verbose: bool = False) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return logging.getLogger(LOGGER_NAME)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: list[str] | None = None) -> pd.DataFrame:
    ensure_dir(path.parent)
    df = pd.DataFrame(list(rows), columns=columns)
    df.to_csv(path, index=False)
    return df


def vec3(value: Any) -> np.ndarray:
    if value is None:
        return np.full(3, np.nan)
    try:
        if hasattr(value, "get"):
            return np.array([float(value.get(i)) for i in range(3)], dtype=float)
        return np.array(list(value), dtype=float)[:3]
    except Exception:
        return np.full(3, np.nan)


def norm_error(a: Any, b: Any) -> float:
    aa = vec3(a)
    bb = vec3(b)
    if np.any(np.isnan(aa)) or np.any(np.isnan(bb)):
        return math.nan
    return float(np.linalg.norm(aa - bb))


def rmse(values: Iterable[float]) -> float:
    arr = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return math.nan
    return float(np.sqrt(np.mean(arr * arr)))


def correlation(a: Iterable[float], b: Iterable[float]) -> float:
    aa = np.array(list(a), dtype=float)
    bb = np.array(list(b), dtype=float)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if mask.sum() < 2:
        return math.nan
    aa = aa[mask]
    bb = bb[mask]
    if np.std(aa) == 0 or np.std(bb) == 0:
        return math.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def rel_error(abs_error: float, reference: float) -> float:
    if not np.isfinite(abs_error) or not np.isfinite(reference) or abs(reference) < 1e-12:
        return math.nan
    return float(abs_error / abs(reference))


def status_from_errors(max_error: float | None, warn: float, fail: float) -> str:
    if max_error is None or not np.isfinite(max_error):
        return "not evaluated"
    if max_error > fail:
        return "failed"
    if max_error > warn:
        return "warning"
    return "passed"
