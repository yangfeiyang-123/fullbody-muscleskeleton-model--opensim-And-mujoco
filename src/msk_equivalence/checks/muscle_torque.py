from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from msk_equivalence.checks.kinematics import _pose_values
from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import rel_error, rmse, write_csv, write_worst_csv


def _plan_rows(mapping: MappingConfig) -> list[dict[str, Any]]:
    experiments = mapping.dynamics_experiments.get("muscle_torque", [])
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
                "activation_values": item.get("activation_values", ""),
                "groups": item.get("groups", ""),
                "seed": item.get("seed", ""),
                "vector_count": item.get("vector_count", ""),
                "active_muscles_per_vector": item.get("active_muscles_per_vector", ""),
                "contacts": item.get("contacts", ""),
                "compare": item.get("compare", ""),
                "purpose": item.get("purpose", ""),
                "status": "planned",
            }
        )
    return rows


def _activation_values(mapping: MappingConfig) -> list[float]:
    experiments = mapping.dynamics_experiments.get("muscle_torque", [])
    for item in experiments if isinstance(experiments, list) else []:
        if isinstance(item, dict) and item.get("name") == "single_muscle_activation_sweep":
            values = item.get("activation_values", [0.0, 1.0])
            if isinstance(values, list):
                return [float(v) for v in values]
    return [0.0, 1.0]


def _calibration_values() -> list[float]:
    return [float(v) for v in np.linspace(0.0, 1.0, 21)]


def _set_opensim_activations(osim: Any, active_muscle: str | None, activation: float) -> None:
    muscles = osim.model.getMuscles()
    for i in range(muscles.getSize()):
        muscle = muscles.get(i)
        try:
            muscle.setActivation(osim.state, activation if muscle.getName() == active_muscle else 0.0)
        except Exception:
            pass


def _opensim_actuation(osim: Any, muscle_name: str, activation: float, pose: dict[str, float]) -> float:
    osim.set_pose(pose)
    _set_opensim_activations(osim, muscle_name, activation)
    muscle = osim.model.getMuscles().get(muscle_name)
    try:
        muscle.computeEquilibrium(osim.state)
    except Exception:
        pass
    try:
        osim.model.realizeDynamics(osim.state)
    except Exception:
        pass
    return float(muscle.getActuation(osim.state))


def _opensim_min_activation(osim: Any, muscle_name: str) -> float:
    try:
        muscle = osim.model.getMuscles().get(muscle_name)
        if hasattr(osim.opensim, "Thelen2003Muscle"):
            thelen = osim.opensim.Thelen2003Muscle.safeDownCast(muscle)
            if thelen:
                return float(thelen.getMinimumActivation())
        if hasattr(muscle, "getMinimumActivation"):
            return float(muscle.getMinimumActivation())
    except Exception:
        pass
    return 0.0


def _mujoco_actuator_force(mjcf: Any, actuator_name: str, activation: float, pose: dict[str, float]) -> float:
    mj = mjcf.mujoco
    model = mjcf.model
    data = mj.MjData(model)
    data.qpos[:] = model.qpos0
    for name, value in pose.items():
        idx = mjcf.qpos_index(name)
        if idx is not None and 0 <= idx < model.nq:
            data.qpos[idx] = float(value)
    data.qvel[:] = 0.0
    if model.nu:
        data.ctrl[:] = 0.0
    if data.act is not None and data.act.size:
        data.act[:] = 0.0
    actuator_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    if actuator_id < 0:
        raise KeyError(actuator_name)
    data.ctrl[actuator_id] = activation
    adr = int(model.actuator_actadr[actuator_id])
    num = int(model.actuator_actnum[actuator_id])
    if adr >= 0 and num > 0:
        data.act[adr : adr + num] = activation
    mj.mj_forward(model, data)
    return abs(float(data.actuator_force[actuator_id]))


def _interp_activation_for_force(curve: list[tuple[float, float]], target_force: float) -> float:
    clean = sorted((force, activation) for activation, force in curve if np.isfinite(force))
    if not clean:
        return 0.0
    if target_force <= clean[0][0]:
        return float(clean[0][1])
    if target_force >= clean[-1][0]:
        return float(clean[-1][1])
    for (f0, a0), (f1, a1) in zip(clean[:-1], clean[1:]):
        if f0 <= target_force <= f1:
            if abs(f1 - f0) < 1e-12:
                return float(a0)
            t = (target_force - f0) / (f1 - f0)
            return float(a0 + t * (a1 - a0))
    return float(clean[-1][1])


def _force_rows(osim: Any, mjcf: Any, mapping: MappingConfig) -> list[dict[str, Any]]:
    sample = next((s for s in mapping.pose_samples if str(s.get("name", "")) == "neutral"), mapping.pose_samples[0])
    opensim_pose = _pose_values(sample, mapping, "opensim")
    mujoco_pose = _pose_values(sample, mapping, "mujoco")
    activation_values = _activation_values(mapping)
    rows = []
    for item in mapping.muscles:
        oname = mapping.side_name(item, "opensim")
        mname = mapping.side_name(item, "mujoco")
        if not oname or not mname:
            continue
        try:
            opensim_baseline = abs(_opensim_actuation(osim, oname, 0.0, opensim_pose))
            mujoco_baseline = _mujoco_actuator_force(mjcf, mname, 0.0, mujoco_pose)
            opensim_curve = [
                (activation, abs(_opensim_actuation(osim, oname, activation, opensim_pose)) - opensim_baseline)
                for activation in _calibration_values()
            ]
        except Exception as exc:
            rows.append(
                {
                    "experiment": "single_muscle_activation_sweep",
                    "opensim_muscle": oname,
                    "mujoco_actuator": mname,
                    "mujoco_activation": "",
                    "opensim_adapter_activation": "",
                    "opensim_active_force_n": np.nan,
                    "mujoco_active_force_n": np.nan,
                    "absolute_error_n": np.nan,
                    "relative_error": np.nan,
                    "status": f"skipped: {exc}",
                }
            )
            continue
        for activation in activation_values:
            try:
                mujoco_force = _mujoco_actuator_force(mjcf, mname, activation, mujoco_pose) - mujoco_baseline
                opensim_activation = _interp_activation_for_force(opensim_curve, mujoco_force)
                opensim_force = abs(_opensim_actuation(osim, oname, opensim_activation, opensim_pose)) - opensim_baseline
                error = abs(opensim_force - mujoco_force)
                status = "evaluated"
            except Exception as exc:
                opensim_activation = opensim_force = mujoco_force = error = np.nan
                status = f"skipped: {exc}"
            rows.append(
                {
                    "experiment": "single_muscle_activation_sweep",
                    "opensim_muscle": oname,
                    "mujoco_actuator": mname,
                    "mujoco_activation": activation,
                    "opensim_adapter_activation": opensim_activation,
                    "opensim_active_force_n": opensim_force,
                    "mujoco_active_force_n": mujoco_force,
                    "absolute_error_n": error,
                    "relative_error": rel_error(error, max(abs(opensim_force), abs(mujoco_force))),
                    "status": status,
                }
            )
    return rows


def _status(df: Any, mapping: MappingConfig) -> str:
    if df.empty:
        return "not evaluated"
    finite = df[df["status"] == "evaluated"].copy()
    finite = finite[np.array(finite["mujoco_activation"], dtype=float) > 0.0] if not finite.empty else finite
    if finite.empty:
        return "not evaluated"
    abs_err = np.array(finite["absolute_error_n"], dtype=float)
    rel_err = np.array(finite["relative_error"], dtype=float)
    reference = np.maximum(np.abs(np.array(finite["opensim_active_force_n"], dtype=float)), np.abs(np.array(finite["mujoco_active_force_n"], dtype=float)))
    fail_abs = float(mapping.thresholds.get("muscle_force_fail_n", 3.0))
    warn_abs = float(mapping.thresholds.get("muscle_force_warning_n", 1.0))
    fail_rel = float(mapping.thresholds.get("muscle_force_fail_rel", 0.10))
    warn_rel = float(mapping.thresholds.get("muscle_force_warning_rel", 0.05))
    rel_floor = float(mapping.thresholds.get("muscle_force_relative_reference_floor_n", warn_abs))
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
        write_csv(out_dir / "muscle_torque_plan.csv", planned)
    rows = _force_rows(osim, mjcf, mapping)
    df = write_csv(out_dir / "muscle_generated_torque_error.csv", rows)
    files = ["muscle_generated_torque_error.csv"]
    worst = write_worst_csv(out_dir / "diagnostics" / "muscle_torque_worst_force_error.csv", df, "absolute_error_n")
    if worst:
        files.append(f"diagnostics/{worst}")
    if planned:
        files.append("muscle_torque_plan.csv")
    finite = df[df["status"] == "evaluated"] if not df.empty else df
    finite_active = finite[np.array(finite["mujoco_activation"], dtype=float) > 0.0] if not finite.empty else finite
    status = _status(df, mapping)
    reason = None
    if status == "failed":
        reason = "Single-muscle active-force sweep mismatch exceeds thresholds; inspect diagnostics/muscle_torque_worst_force_error.csv."
    return {
        "status": status,
        **({"reason": reason} if reason else {}),
        "planned_experiments": len(planned),
        "evaluated_rows": int(len(finite_active)) if not finite_active.empty else 0,
        "max_abs_error_n": float(finite_active["absolute_error_n"].max()) if not finite_active.empty else None,
        "rmse_abs_error_n": rmse(finite_active["absolute_error_n"]) if not finite_active.empty else None,
        "warning_threshold_n": float(mapping.thresholds.get("muscle_force_warning_n", 1.0)),
        "failure_threshold_n": float(mapping.thresholds.get("muscle_force_fail_n", 3.0)),
        "note": "This executable gate compares neutral-pose single-muscle active force after fitting an OpenSim activation adapter for MuJoCo controls. Generalized muscle-torque vector and group/random activation gates are still planned.",
        "files": files,
    }
