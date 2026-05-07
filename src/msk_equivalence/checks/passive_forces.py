from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from msk_equivalence.checks.kinematics import _pose_values
from msk_equivalence.checks.muscle_torque import _mujoco_actuator_force, _opensim_actuation, _opensim_min_activation
from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import rel_error, rmse, write_csv, write_worst_csv


def _plan_rows(mapping: MappingConfig) -> list[dict[str, Any]]:
    experiments = mapping.dynamics_experiments.get("passive_forces", [])
    if not isinstance(experiments, list):
        return []
    rows = []
    for item in experiments:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "name": item.get("name", ""),
                "q_samples": item.get("q_samples", ""),
                "qdot": item.get("qdot", ""),
                "qdot_values": item.get("qdot_values", ""),
                "activation": item.get("activation", ""),
                "contacts": item.get("contacts", ""),
                "compare": item.get("compare", ""),
                "purpose": item.get("purpose", ""),
                "status": "planned",
            }
        )
    return rows


def _passive_rows(osim: Any, mjcf: Any, mapping: MappingConfig) -> list[dict[str, Any]]:
    sample = next((s for s in mapping.pose_samples if str(s.get("name", "")) == "neutral"), mapping.pose_samples[0])
    opensim_pose = _pose_values(sample, mapping, "opensim")
    mujoco_pose = _pose_values(sample, mapping, "mujoco")
    rows: list[dict[str, Any]] = []
    for item in mapping.muscles:
        oname = mapping.side_name(item, "opensim")
        mname = mapping.side_name(item, "mujoco")
        if not oname or not mname:
            continue
        try:
            activation_floor = _opensim_min_activation(osim, oname)
            opensim_force = abs(_opensim_actuation(osim, oname, 0.0, opensim_pose))
            mujoco_force = _mujoco_actuator_force(mjcf, mname, activation_floor, mujoco_pose)
            mujoco_active_capacity = max(0.0, _mujoco_actuator_force(mjcf, mname, 1.0, mujoco_pose) - mujoco_force)
            error = abs(opensim_force - mujoco_force)
            status = "evaluated"
        except Exception as exc:
            activation_floor = opensim_force = mujoco_force = mujoco_active_capacity = error = np.nan
            status = f"skipped: {exc}"
        rows.append(
            {
                "experiment": "neutral_zero_activation_passive_muscle_force",
                "opensim_muscle": oname,
                "mujoco_actuator": mname,
                "activation_floor": activation_floor,
                "opensim_passive_force_n": opensim_force,
                "mujoco_passive_force_n": mujoco_force,
                "mujoco_active_force_capacity_n": mujoco_active_capacity,
                "absolute_error_n": error,
                "relative_error": rel_error(error, max(abs(opensim_force), abs(mujoco_force))),
                "relative_error_vs_active_capacity": rel_error(error, mujoco_active_capacity),
                "status": status,
            }
        )
    return rows


def _status(df: Any, mapping: MappingConfig) -> str:
    if df.empty:
        return "not evaluated"
    finite = df[df["status"] == "evaluated"].copy()
    if finite.empty:
        return "not evaluated"
    abs_err = np.array(finite["absolute_error_n"], dtype=float)
    reference = np.maximum(
        np.abs(np.array(finite["opensim_passive_force_n"], dtype=float)),
        np.abs(np.array(finite["mujoco_passive_force_n"], dtype=float)),
    )
    if bool(mapping.thresholds.get("passive_force_use_active_reference", True)) and "mujoco_active_force_capacity_n" in finite:
        reference = np.maximum(reference, np.abs(np.array(finite["mujoco_active_force_capacity_n"], dtype=float)))
    fail_abs = float(mapping.thresholds.get("passive_force_fail_n", 3.0))
    warn_abs = float(mapping.thresholds.get("passive_force_warning_n", 1.0))
    fail_rel = float(mapping.thresholds.get("passive_force_fail_rel", 0.10))
    warn_rel = float(mapping.thresholds.get("passive_force_warning_rel", 0.05))
    rel_floor = float(mapping.thresholds.get("passive_force_relative_reference_floor_n", warn_abs))
    fail_limit = np.maximum(fail_abs, fail_rel * reference)
    warn_limit = np.maximum(warn_abs, warn_rel * reference)
    gated = reference >= rel_floor
    if np.any(abs_err > fail_limit) or np.any((~gated) & (abs_err > fail_abs)):
        return "failed"
    if np.any(abs_err > warn_limit) or np.any((~gated) & (abs_err > warn_abs)):
        return "warning"
    return "passed"


def run(osim: Any, mjcf: Any, mapping: MappingConfig, out_dir: Path) -> dict[str, Any]:
    planned = _plan_rows(mapping)
    if planned:
        write_csv(out_dir / "passive_force_plan.csv", planned)
    rows = _passive_rows(osim, mjcf, mapping)
    df = write_csv(out_dir / "passive_force_comparison.csv", rows)
    write_csv(out_dir / "ligament_comparison.csv", [{"status": "not evaluated", "reason": "No mapped OpenSim ligament force adapter is configured."}])
    files = ["passive_force_comparison.csv", "ligament_comparison.csv"]
    worst = write_worst_csv(out_dir / "diagnostics" / "passive_force_worst_error.csv", df, "absolute_error_n")
    if worst:
        files.append(f"diagnostics/{worst}")
    if planned:
        files.append("passive_force_plan.csv")
    finite = df[df["status"] == "evaluated"] if not df.empty else df
    status = _status(df, mapping)
    reason = None
    if status == "failed":
        reason = "Neutral zero-activation passive muscle force mismatch exceeds thresholds; inspect diagnostics/passive_force_worst_error.csv."
    elif status == "not evaluated":
        reason = "No evaluated passive muscle force rows."
    return {
        "status": status,
        **({"reason": reason} if reason else {}),
        "planned_experiments": len(planned),
        "evaluated_rows": int(len(finite)) if not finite.empty else 0,
        "max_abs_error_n": float(finite["absolute_error_n"].max()) if not finite.empty else None,
        "rmse_abs_error_n": rmse(finite["absolute_error_n"]) if not finite.empty else None,
        "warning_threshold_n": float(mapping.thresholds.get("passive_force_warning_n", 1.0)),
        "failure_threshold_n": float(mapping.thresholds.get("passive_force_fail_n", 3.0)),
        "note": "This executable gate compares neutral-pose zero-command muscle force using the OpenSim Thelen minimum activation as the MuJoCo activation floor. Ligaments, joint damping/limits and passive velocity sweeps remain planned diagnostics.",
        "files": files,
    }
