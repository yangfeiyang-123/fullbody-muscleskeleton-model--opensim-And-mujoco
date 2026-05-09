from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import ensure_dir, rel_error


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read adapter config files.") from exc
    if not path.exists():
        raise FileNotFoundError(f"Adapter config does not exist: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Adapter config must be a YAML mapping: {path}")
    return data


def load_adapter_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return _load_yaml(path)


def adapter_config(mapping: MappingConfig) -> dict[str, Any]:
    value = mapping.raw.get("_adapter_config", {})
    return value if isinstance(value, dict) else {}


def adapter_thresholds(mapping: MappingConfig) -> dict[str, float]:
    cfg = adapter_config(mapping)
    raw = cfg.get("thresholds", {}) if isinstance(cfg.get("thresholds", {}), dict) else {}
    return {
        "qacc_warning": float(raw.get("qacc_warning", mapping.thresholds.get("forward_dynamics_accel_warning", 5.0))),
        "qacc_fail": float(raw.get("qacc_fail", mapping.thresholds.get("forward_dynamics_accel_fail", 20.0))),
        "qacc_warning_rel": float(raw.get("qacc_warning_rel", mapping.thresholds.get("forward_dynamics_accel_warning_rel", 0.05))),
        "qacc_fail_rel": float(raw.get("qacc_fail_rel", mapping.thresholds.get("forward_dynamics_accel_fail_rel", 0.10))),
        "qacc_relative_abs_error_floor": float(raw.get("qacc_relative_abs_error_floor", raw.get("qacc_warning", mapping.thresholds.get("forward_dynamics_accel_warning", 5.0)))),
        "rollout_q_warning": float(raw.get("rollout_q_warning", mapping.thresholds.get("forward_rollout_q_warning", 0.01))),
        "rollout_q_fail": float(raw.get("rollout_q_fail", mapping.thresholds.get("forward_rollout_q_fail", 0.05))),
        "rollout_qdot_warning": float(raw.get("rollout_qdot_warning", mapping.thresholds.get("forward_rollout_qdot_warning", 0.5))),
        "rollout_qdot_fail": float(raw.get("rollout_qdot_fail", mapping.thresholds.get("forward_rollout_qdot_fail", 2.0))),
    }


def split_indices(count: int, mapping: MappingConfig) -> dict[str, list[int]]:
    cfg = adapter_config(mapping)
    split_cfg = cfg.get("split", {}) if isinstance(cfg.get("split", {}), dict) else {}
    explicit = {name: split_cfg.get(f"{name}_indices") for name in ("train", "val", "test")}
    if all(isinstance(v, list) for v in explicit.values()):
        return {
            name: sorted({int(idx) for idx in values if 0 <= int(idx) < count})
            for name, values in explicit.items()
        }
    random_cfg = cfg.get("random_activation", {}) if isinstance(cfg.get("random_activation", {}), dict) else {}
    basis_count = len(actuator_names(mapping)) if bool(random_cfg.get("include_single_muscle_basis", False)) else 0
    basis_count = min(basis_count, count)
    seed = int(split_cfg.get("seed", cfg.get("seed", 20260509)))
    train_fraction = float(split_cfg.get("train_fraction", 0.65))
    val_fraction = float(split_cfg.get("val_fraction", 0.15))
    if count <= 0:
        return {"train": [], "val": [], "test": []}
    rng = np.random.default_rng(seed)
    indices = np.arange(basis_count, count)
    rng.shuffle(indices)
    random_count = count - basis_count
    if random_count >= 4:
        train_count = max(1, min(random_count - 2, int(round(random_count * train_fraction))))
        val_count = max(1, min(random_count - train_count - 1, int(round(random_count * val_fraction))))
    elif random_count == 3:
        train_count, val_count = 1, 1
    elif random_count == 2:
        train_count, val_count = 1, 0
    else:
        train_count, val_count = random_count, 0
    train = sorted([*range(basis_count), *(int(i) for i in indices[:train_count])])
    val = sorted(int(i) for i in indices[train_count : train_count + val_count])
    test = sorted(int(i) for i in indices[train_count + val_count :])
    return {"train": train, "val": val, "test": test}


def actuator_names(mapping: MappingConfig) -> list[str]:
    names = [mapping.side_name(item, "mujoco") for item in mapping.muscles]
    return sorted({str(name) for name in names if name})


def feature_vector(activations: dict[str, float], names: list[str]) -> np.ndarray:
    values = [1.0]
    values.extend(float(activations.get(name, 0.0)) for name in names)
    values.append(float(sum(1 for value in activations.values() if abs(float(value)) > 1e-12)))
    return np.array(values, dtype=float)


def feature_matrix(vectors: list[dict[str, float]], names: list[str], indices: list[int]) -> np.ndarray:
    return np.vstack([feature_vector(vectors[idx], names) for idx in indices]) if indices else np.zeros((0, len(names) + 2), dtype=float)


@dataclass(frozen=True)
class ResidualLinearAdapter:
    actuator_names: list[str]
    coordinate_names: list[str]
    weights: np.ndarray
    regularization: float

    def predict(self, activations: dict[str, float]) -> np.ndarray:
        return feature_vector(activations, self.actuator_names) @ self.weights

    def save_npz(self, path: Path) -> None:
        ensure_dir(path.parent)
        np.savez(
            path,
            actuator_names=np.array(self.actuator_names, dtype=object),
            coordinate_names=np.array(self.coordinate_names, dtype=object),
            weights=self.weights,
            regularization=np.array([self.regularization], dtype=float),
        )


def fit_residual_adapter(
    vectors: list[dict[str, float]],
    coordinate_names: list[str],
    residuals_by_vector: dict[int, np.ndarray],
    train_indices: list[int],
    mapping: MappingConfig,
) -> ResidualLinearAdapter | None:
    if not train_indices:
        return None
    names = actuator_names(mapping)
    x = feature_matrix(vectors, names, train_indices)
    y = np.vstack([residuals_by_vector[idx] for idx in train_indices])
    if x.size == 0 or y.size == 0:
        return None
    cfg = adapter_config(mapping)
    reg = float(cfg.get("ridge_regularization", 1e-3))
    penalty = np.eye(x.shape[1], dtype=float) * reg
    penalty[0, 0] = 0.0
    try:
        weights = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(x.T @ x + penalty, rcond=1e-12) @ x.T @ y
    return ResidualLinearAdapter(names, coordinate_names, weights, reg)


def qacc_status(rows: Any, mapping: MappingConfig, *, split: str | None = None) -> str:
    if rows.empty:
        return "not evaluated"
    finite = rows[(rows["status"] == "evaluated") & (rows["coordinate_role"] == "independent")].copy()
    if split is not None and "split" in finite:
        finite = finite[finite["split"] == split]
    if finite.empty:
        return "not evaluated"
    thresholds = adapter_thresholds(mapping)
    abs_err = np.array(finite["absolute_error"], dtype=float)
    rel_err = np.array(finite["relative_error"], dtype=float)
    rel_floor = thresholds["qacc_relative_abs_error_floor"]
    finite_rel = rel_err[np.isfinite(rel_err) & (abs_err >= rel_floor)]
    max_abs = float(np.nanmax(abs_err))
    max_rel = float(np.nanmax(finite_rel)) if finite_rel.size else 0.0
    if max_abs > thresholds["qacc_fail"] or max_rel > thresholds["qacc_fail_rel"]:
        return "failed"
    if max_abs > thresholds["qacc_warning"] or max_rel > thresholds["qacc_warning_rel"]:
        return "warning"
    return "passed"


def rollout_status(rows: Any, mapping: MappingConfig) -> str:
    if rows.empty:
        return "not evaluated"
    finite = rows[rows["status"] == "evaluated"].copy()
    if finite.empty:
        return "not evaluated"
    thresholds = adapter_thresholds(mapping)
    q_err = np.array(finite["q_error"], dtype=float)
    qdot_err = np.array(finite["qdot_error"], dtype=float)
    if float(np.nanmax(q_err)) > thresholds["rollout_q_fail"] or float(np.nanmax(qdot_err)) > thresholds["rollout_qdot_fail"]:
        return "failed"
    if float(np.nanmax(q_err)) > thresholds["rollout_q_warning"] or float(np.nanmax(qdot_err)) > thresholds["rollout_qdot_warning"]:
        return "warning"
    return "passed"


def coordinate_group(name: str) -> str:
    lowered = name.lower()
    if any(key in lowered for key in ("cmc", "mp_", "mpthumb", "ip_flexion", "mcp", "pm", "md")):
        return "hand_thumb_finger"
    if any(key in lowered for key in ("wrist", "pro_sup")):
        return "wrist_forearm"
    if any(key in lowered for key in ("ankle", "knee", "hip", "subtalar", "mtp")):
        return "lower_limb"
    if any(key in lowered for key in ("elv", "shoulder", "elbow")):
        return "upper_limb"
    if any(key in lowered for key in ("pelvis", "flex_extension", "lat_bending", "rotation")):
        return "root_torso"
    return "other"


def row_relative_error(predicted: float, target: float) -> float:
    return rel_error(abs(predicted - target), max(abs(predicted), abs(target)))
