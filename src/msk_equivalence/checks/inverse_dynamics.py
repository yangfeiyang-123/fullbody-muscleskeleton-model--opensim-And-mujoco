from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np

from msk_equivalence.checks.kinematics import _pose_values
from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import rel_error, rmse, write_csv, write_worst_csv


ROOT_TRANSLATION_PAIRS = [
    ("root_tx", "qpos[0]", "x"),
    ("root_ty", "qpos[1]", "y"),
    ("root_tz", "qpos[2]", "z"),
]


def _rows(mapping: MappingConfig) -> list[dict[str, Any]]:
    experiments = mapping.dynamics_experiments.get("inverse_dynamics", [])
    if not isinstance(experiments, list):
        return []
    rows = []
    for item in experiments:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "name": item.get("name", ""),
                "q_sample": item.get("q_sample", ""),
                "q_trajectory": item.get("q_trajectory", ""),
                "qdot": item.get("qdot", ""),
                "qddot": item.get("qddot", ""),
                "contacts": item.get("contacts", ""),
                "compare": item.get("compare", ""),
                "purpose": item.get("purpose", ""),
                "status": "planned",
            }
        )
    return rows


def _dependent_coordinates(osim: Any) -> set[str]:
    try:
        from msk_equivalence.checks.moment_arm import _dependent_coordinates as helper

        return helper(osim)
    except Exception:
        return set()


def _coupler_constraints(osim: Any) -> list[dict[str, Any]]:
    path = getattr(osim, "path", None)
    if path is None:
        return []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except Exception:
        return []
    pattern = re.compile(
        r"<CoordinateCouplerConstraint\b.*?"
        r"<coefficients>(?P<coefficients>.*?)</coefficients>.*?"
        r"<independent_coordinate_names>(?P<independent>.*?)</independent_coordinate_names>.*?"
        r"<dependent_coordinate_name>(?P<dependent>.*?)</dependent_coordinate_name>.*?"
        r"<scale_factor>(?P<scale>.*?)</scale_factor>",
        re.DOTALL,
    )
    constraints: list[dict[str, Any]] = []
    for match in pattern.finditer(text):
        independent = match.group("independent").strip().split()
        if len(independent) != 1:
            continue
        coefficients = [float(x) for x in match.group("coefficients").strip().split()]
        constraints.append(
            {
                "independent": independent[0],
                "dependent": match.group("dependent").strip(),
                "coefficients": coefficients,
                "scale": float(match.group("scale").strip()),
            }
        )
    return constraints


def _eval_opensim_polynomial(coefficients: list[float], x: float) -> float:
    # OpenSim PolynomialFunction stores coefficients highest-order first.
    value = 0.0
    for coefficient in coefficients:
        value = value * x + coefficient
    return value


def _coordinate_targets(mapping: MappingConfig, side: str) -> dict[str, str]:
    targets: dict[str, str] = {}
    for item in mapping.coordinates:
        oname = mapping.side_name(item, "opensim")
        sname = mapping.side_name(item, side)
        if oname and sname:
            targets[oname] = sname
    return targets


def _with_coupled_coordinates(values: dict[str, float], constraints: list[dict[str, Any]], mapping: MappingConfig, side: str) -> dict[str, float]:
    result = dict(values)
    targets = _coordinate_targets(mapping, side)
    changed = True
    while changed:
        changed = False
        for constraint in constraints:
            independent = targets.get(str(constraint["independent"]))
            dependent = targets.get(str(constraint["dependent"]))
            if not independent or not dependent or independent not in result:
                continue
            value = float(constraint["scale"]) * _eval_opensim_polynomial(list(constraint["coefficients"]), float(result[independent]))
            if dependent not in result or abs(float(result[dependent]) - value) > 1e-14:
                result[dependent] = value
                changed = True
    return result


def _opensim_gravity_potential(osim: Any, values: dict[str, float], comparison_gravity: np.ndarray) -> float:
    osim.set_pose(values)
    return -float(osim.total_mass() * np.dot(comparison_gravity, osim.whole_body_com()))


def _mujoco_gravity_potential(mjcf: Any, values: dict[str, float]) -> float:
    mjcf.set_pose(values)
    gravity = np.array(mjcf.model.opt.gravity, dtype=float)
    return -float(mjcf.total_mass() * np.dot(gravity, mjcf.whole_body_com()))


def _opensim_locked(osim: Any, coordinate_name: str) -> bool:
    try:
        return bool(osim.model.getCoordinateSet().get(coordinate_name).getLocked(osim.state))
    except Exception:
        return False


def _opensim_coordinate_gradient(osim: Any, base_values: dict[str, float], coordinate_name: str, eps: float, constraints: list[dict[str, Any]], mapping: MappingConfig, comparison_gravity: np.ndarray) -> float:
    plus = dict(base_values)
    minus = dict(base_values)
    base = float(plus.get(coordinate_name, osim.default_coordinates.get(coordinate_name, 0.0)))
    plus[coordinate_name] = base + eps
    minus[coordinate_name] = base - eps
    plus = _with_coupled_coordinates(plus, constraints, mapping, "opensim")
    minus = _with_coupled_coordinates(minus, constraints, mapping, "opensim")
    return (_opensim_gravity_potential(osim, plus, comparison_gravity) - _opensim_gravity_potential(osim, minus, comparison_gravity)) / (2.0 * eps)


def _mujoco_coordinate_gradient(mjcf: Any, base_values: dict[str, float], coordinate_name: str, eps: float, constraints: list[dict[str, Any]], mapping: MappingConfig) -> float:
    plus = dict(base_values)
    minus = dict(base_values)
    base = float(plus.get(coordinate_name, 0.0))
    plus[coordinate_name] = base + eps
    minus[coordinate_name] = base - eps
    plus = _with_coupled_coordinates(plus, constraints, mapping, "mujoco")
    minus = _with_coupled_coordinates(minus, constraints, mapping, "mujoco")
    return (_mujoco_gravity_potential(mjcf, plus) - _mujoco_gravity_potential(mjcf, minus)) / (2.0 * eps)


def _comparison_frame_root_rows(osim: Any, mjcf: Any, mapping: MappingConfig, base_osim: dict[str, float], base_mj: dict[str, float], eps: float, constraints: list[dict[str, Any]], comparison_gravity: np.ndarray) -> list[dict[str, Any]]:
    try:
        osim_root = np.array([_opensim_coordinate_gradient(osim, base_osim, oname, eps, constraints, mapping, comparison_gravity) for oname, _, _ in ROOT_TRANSLATION_PAIRS], dtype=float)
        mujoco_root = np.array([_mujoco_coordinate_gradient(mjcf, base_mj, mname, eps, constraints, mapping) for _, mname, _ in ROOT_TRANSLATION_PAIRS], dtype=float)
    except Exception as exc:
        return [
            {
                "experiment": "gravity_static_neutral",
                "method": "comparison_frame_root_translation_gradient",
                "opensim_coordinate": "root_translation_vector",
                "mujoco_qpos_or_joint": "freejoint_translation_vector",
                "coordinate_role": "independent",
                "opensim_tau_nm": np.nan,
                "mujoco_tau_nm": np.nan,
                "absolute_error_nm": np.nan,
                "relative_error": np.nan,
                "sign_consistent": None,
                "status": f"skipped: {exc}",
            }
        ]
    rows = []
    for idx, axis in enumerate(["x", "y", "z"]):
        otau = float(osim_root[idx])
        mtau = float(mujoco_root[idx])
        err = abs(otau - mtau)
        rows.append(
            {
                "experiment": "gravity_static_neutral",
                "method": "comparison_frame_root_translation_gradient",
                "opensim_coordinate": f"root_translation_{axis}_in_mujoco_frame",
                "mujoco_qpos_or_joint": f"freejoint_translation_{axis}",
                "coordinate_role": "independent",
                "opensim_tau_nm": otau,
                "mujoco_tau_nm": mtau,
                "absolute_error_nm": err,
                "relative_error": rel_error(err, otau),
                "sign_consistent": bool(np.sign(otau) == np.sign(mtau) or abs(otau) < 1e-9 or abs(mtau) < 1e-9),
                "status": "evaluated",
            }
        )
    return rows


def _potential_gradient_rows(osim: Any, mjcf: Any, mapping: MappingConfig) -> list[dict[str, Any]]:
    sample = next((s for s in mapping.pose_samples if str(s.get("name", "")) == "neutral"), mapping.pose_samples[0])
    base_osim = _pose_values(sample, mapping, "opensim")
    base_mj = _pose_values(sample, mapping, "mujoco")
    dependent = _dependent_coordinates(osim)
    constraints = _coupler_constraints(osim)
    base_osim = _with_coupled_coordinates(base_osim, constraints, mapping, "opensim")
    base_mj = _with_coupled_coordinates(base_mj, constraints, mapping, "mujoco")
    comparison_gravity = np.array(mjcf.model.opt.gravity, dtype=float)
    eps = float(mapping.thresholds.get("inverse_dynamics_fd_epsilon", 1e-6))
    rows: list[dict[str, Any]] = _comparison_frame_root_rows(osim, mjcf, mapping, base_osim, base_mj, eps, constraints, comparison_gravity)
    root_translation_names = {oname for oname, _, _ in ROOT_TRANSLATION_PAIRS}
    for item in mapping.coordinates:
        oname = mapping.side_name(item, "opensim")
        mname = mapping.side_name(item, "mujoco")
        if not oname or not mname:
            continue
        role = "dependent" if oname in dependent else "independent"
        if oname in root_translation_names:
            rows.append(
                {
                    "experiment": "gravity_static_neutral",
                    "method": "native_root_translation_gradient",
                    "opensim_coordinate": oname,
                    "mujoco_qpos_or_joint": mname,
                    "coordinate_role": "root_translation_component",
                    "opensim_tau_nm": _opensim_coordinate_gradient(osim, base_osim, oname, eps, constraints, mapping, comparison_gravity),
                    "mujoco_tau_nm": _mujoco_coordinate_gradient(mjcf, base_mj, mname, eps, constraints, mapping),
                    "absolute_error_nm": np.nan,
                    "relative_error": np.nan,
                    "sign_consistent": None,
                    "status": "diagnostic: compared through comparison_frame_root_translation_gradient",
                }
            )
            continue
        if mname == "root":
            rows.append(
                {
                    "experiment": "gravity_static_neutral",
                    "method": "gravity_potential_gradient",
                    "opensim_coordinate": oname,
                    "mujoco_qpos_or_joint": mname,
                    "coordinate_role": role,
                    "opensim_tau_nm": np.nan,
                    "mujoco_tau_nm": np.nan,
                    "absolute_error_nm": np.nan,
                    "relative_error": np.nan,
                    "sign_consistent": None,
                    "status": "skipped: root Euler coordinate requires freejoint quaternion torque adapter",
                }
            )
            continue
        if _opensim_locked(osim, oname):
            rows.append(
                {
                    "experiment": "gravity_static_neutral",
                    "method": "gravity_potential_gradient",
                    "opensim_coordinate": oname,
                    "mujoco_qpos_or_joint": mname,
                    "coordinate_role": "locked",
                    "opensim_tau_nm": np.nan,
                    "mujoco_tau_nm": np.nan,
                    "absolute_error_nm": np.nan,
                    "relative_error": np.nan,
                    "sign_consistent": None,
                    "status": "skipped: OpenSim coordinate is locked",
                }
            )
            continue
        try:
            otau = _opensim_coordinate_gradient(osim, base_osim, oname, eps, constraints, mapping, comparison_gravity)
            sign = -1.0 if bool(item.get("sign_flip", False)) else 1.0
            mtau = sign * _mujoco_coordinate_gradient(mjcf, base_mj, mname, eps, constraints, mapping)
            err = abs(otau - mtau)
            sign_consistent = bool(np.sign(otau) == np.sign(mtau) or abs(otau) < 1e-9 or abs(mtau) < 1e-9)
            status = "evaluated"
        except Exception as exc:
            otau = mtau = err = np.nan
            sign_consistent = None
            status = f"skipped: {exc}"
        rows.append(
            {
                "experiment": "gravity_static_neutral",
                "method": "gravity_potential_gradient",
                "opensim_coordinate": oname,
                "mujoco_qpos_or_joint": mname,
                "coordinate_role": role,
                "opensim_tau_nm": otau,
                "mujoco_tau_nm": mtau,
                "absolute_error_nm": err,
                "relative_error": rel_error(err, otau),
                "sign_consistent": sign_consistent,
                "status": status,
            }
        )
    return rows


def _static_gravity_rows(osim: Any, mjcf: Any, mapping: MappingConfig) -> list[dict[str, Any]]:
    # Use the potential-gradient form for the static gravity gate. It is the
    # qdot=0, qddot=0 inverse-dynamics gravity term, while avoiding backend-
    # specific constraint force allocation in floating-base models.
    return _potential_gradient_rows(osim, mjcf, mapping)


def _status(df: Any, mapping: MappingConfig) -> str:
    if df.empty:
        return "not evaluated"
    finite = df[(df["status"] == "evaluated") & (df["coordinate_role"] == "independent")].copy()
    if finite.empty:
        return "not evaluated"
    abs_err = np.array(finite["absolute_error_nm"], dtype=float)
    rel_err = np.array(finite["relative_error"], dtype=float)
    reference = np.maximum(np.abs(np.array(finite["opensim_tau_nm"], dtype=float)), np.abs(np.array(finite["mujoco_tau_nm"], dtype=float)))
    fail_abs = float(mapping.thresholds.get("inverse_dynamics_fail_nm", 3.0))
    warn_abs = float(mapping.thresholds.get("inverse_dynamics_warning_nm", 1.0))
    fail_rel = float(mapping.thresholds.get("inverse_dynamics_fail_rel", 0.05))
    warn_rel = float(mapping.thresholds.get("inverse_dynamics_warning_rel", 0.02))
    rel_floor = float(mapping.thresholds.get("inverse_dynamics_relative_reference_floor_nm", warn_abs))
    finite_rel = rel_err[np.isfinite(rel_err) & (reference >= rel_floor)]
    if np.nanmax(abs_err) > fail_abs or (finite_rel.size and np.nanmax(finite_rel) > fail_rel):
        return "failed"
    if np.nanmax(abs_err) > warn_abs or (finite_rel.size and np.nanmax(finite_rel) > warn_rel):
        return "warning"
    return "passed"


def run(osim: Any, mjcf: Any, mapping: MappingConfig, out_dir: Path) -> dict[str, Any]:
    motion = mapping.raw.get("motion_data", {})
    planned = _rows(mapping)
    if planned:
        write_csv(out_dir / "inverse_dynamics_plan.csv", planned)
    rows = _static_gravity_rows(osim, mjcf, mapping)
    if motion:
        rows.append(
            {
                "experiment": "sagittal_sinusoid_no_contact",
                "status": "not evaluated",
                "reason": "Trajectory inverse dynamics adapter is not implemented yet.",
            }
        )
    else:
        rows.append(
            {
                "experiment": "sagittal_sinusoid_no_contact",
                "status": "not evaluated",
                "reason": "No q(t), qdot(t), qddot(t) motion data adapter configured yet.",
            }
        )
    df = write_csv(out_dir / "inverse_dynamics_torque_error.csv", rows)
    files = ["inverse_dynamics_torque_error.csv"]
    worst = write_worst_csv(out_dir / "diagnostics" / "inverse_dynamics_worst_error.csv", df, "absolute_error_nm")
    if worst:
        files.append(f"diagnostics/{worst}")
    if planned:
        files.append("inverse_dynamics_plan.csv")
    finite = df[(df.get("status") == "evaluated") & (df.get("coordinate_role") == "independent")] if not df.empty else df
    status = _status(df, mapping)
    reason = None
    if status == "failed":
        reason = "Static gravity inverse dynamics mismatch exceeds thresholds; inspect diagnostics/inverse_dynamics_worst_error.csv."
    elif status == "not evaluated":
        reason = "No evaluated independent-coordinate inverse dynamics rows."
    result = {
        "status": status,
        "planned_experiments": len(planned),
        "evaluated_rows": int(len(finite)) if not finite.empty else 0,
        "max_abs_error_nm": float(finite["absolute_error_nm"].max()) if not finite.empty else None,
        "rmse_abs_error_nm": rmse(finite["absolute_error_nm"]) if not finite.empty else None,
        "warning_threshold_nm": float(mapping.thresholds.get("inverse_dynamics_warning_nm", 1.0)),
        "failure_threshold_nm": float(mapping.thresholds.get("inverse_dynamics_fail_nm", 3.0)),
        "files": files,
    }
    if reason:
        result["reason"] = reason
    return result
