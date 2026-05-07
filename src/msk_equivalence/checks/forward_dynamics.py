from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from msk_equivalence.checks.kinematics import _pose_values
from msk_equivalence.checks.inverse_dynamics import _coupler_constraints, _dependent_coordinates, _with_coupled_coordinates
from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import rel_error, rmse, write_csv, write_markdown, write_worst_csv


def _plan_rows(mapping: MappingConfig) -> list[dict[str, Any]]:
    experiments = mapping.dynamics_experiments.get("forward_dynamics", [])
    if not isinstance(experiments, list):
        return []
    rows = []
    for item in experiments:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "name": item.get("name", ""),
                "horizon_s": item.get("horizon_s", ""),
                "dt_s": item.get("dt_s", ""),
                "q_sample": item.get("q_sample", ""),
                "qdot": item.get("qdot", ""),
                "activation": item.get("activation", ""),
                "input_profile": item.get("input_profile", ""),
                "activation_profile": item.get("activation_profile", ""),
                "contact_probe_grid": item.get("contact_probe_grid", ""),
                "torque_coordinates": item.get("torque_coordinates", ""),
                "contacts": item.get("contacts", ""),
                "compare": item.get("compare", ""),
                "purpose": item.get("purpose", ""),
                "status": "planned",
            }
        )
    return rows


def _qvel_index(mjcf: Any, name: str) -> int | None:
    if name.startswith("qpos[") and name.endswith("]"):
        idx = int(name[5:-1])
        return idx if 0 <= idx < int(mjcf.model.nv) else None
    joint_id = mjcf.mujoco.mj_name2id(mjcf.model, mjcf.mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        return None
    return int(mjcf.model.jnt_dofadr[joint_id])


def _set_opensim_zero_state(osim: Any) -> None:
    coord_set = osim.model.getCoordinateSet()
    for i in range(coord_set.getSize()):
        try:
            coord_set.get(i).setSpeedValue(osim.state, 0.0)
        except Exception:
            continue
    muscles = osim.model.getMuscles()
    for i in range(muscles.getSize()):
        muscle = muscles.get(i)
        try:
            muscle.setActivation(osim.state, 0.0)
        except Exception:
            pass
        try:
            muscle.computeEquilibrium(osim.state)
        except Exception:
            pass


def _mujoco_forward_zero_control(mjcf: Any, pose: dict[str, float], disable_contact: bool) -> None:
    mj = mjcf.mujoco
    model = mjcf.model
    data = mjcf.data
    old_disable = int(model.opt.disableflags)
    try:
        if disable_contact:
            model.opt.disableflags = old_disable | int(mj.mjtDisableBit.mjDSBL_CONTACT)
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
        mj.mj_forward(model, data)
    finally:
        model.opt.disableflags = old_disable


def _opensim_locked(osim: Any, coordinate_name: str) -> bool:
    try:
        return bool(osim.model.getCoordinateSet().get(coordinate_name).getLocked(osim.state))
    except Exception:
        return False


def _instant_acceleration_rows(osim: Any, mjcf: Any, mapping: MappingConfig) -> list[dict[str, Any]]:
    sample = next((s for s in mapping.pose_samples if str(s.get("name", "")) == "neutral"), mapping.pose_samples[0])
    constraints = _coupler_constraints(osim)
    opensim_pose = _with_coupled_coordinates(_pose_values(sample, mapping, "opensim"), constraints, mapping, "opensim")
    mujoco_pose = _with_coupled_coordinates(_pose_values(sample, mapping, "mujoco"), constraints, mapping, "mujoco")
    dependent = _dependent_coordinates(osim)

    osim.set_pose(opensim_pose)
    _set_opensim_zero_state(osim)
    try:
        osim.model.realizeAcceleration(osim.state)
    except Exception as exc:
        return [
            {
                "experiment": "passive_no_contact_instant_acceleration",
                "opensim_coordinate": "",
                "mujoco_qvel": "",
                "coordinate_role": "",
                "opensim_qacc": np.nan,
                "mujoco_qacc": np.nan,
                "absolute_error": np.nan,
                "relative_error": np.nan,
                "status": f"skipped: OpenSim acceleration realization failed: {exc}",
            }
        ]

    _mujoco_forward_zero_control(mjcf, mujoco_pose, disable_contact=True)
    coord_set = osim.model.getCoordinateSet()
    rows: list[dict[str, Any]] = []
    for item in mapping.coordinates:
        oname = mapping.side_name(item, "opensim")
        mname = mapping.side_name(item, "mujoco")
        if not oname or not mname:
            continue
        role = "dependent" if oname in dependent else "independent"
        if role == "independent" and _opensim_locked(osim, oname):
            role = "locked"
        if role != "independent" or oname.startswith("root_") or mname == "root":
            rows.append(
                {
                    "experiment": "passive_no_contact_instant_acceleration",
                    "opensim_coordinate": oname,
                    "mujoco_qvel": mname,
                    "coordinate_role": role if role != "independent" else "root_or_freejoint",
                    "opensim_qacc": np.nan,
                    "mujoco_qacc": np.nan,
                    "absolute_error": np.nan,
                    "relative_error": np.nan,
                    "status": "skipped: not a directly comparable independent scalar coordinate",
                }
            )
            continue
        try:
            qvel_idx = _qvel_index(mjcf, mname)
            if qvel_idx is None:
                raise KeyError(f"MuJoCo qvel index not found for {mname}")
            opensim_qacc = float(coord_set.get(oname).getAccelerationValue(osim.state))
            sign = -1.0 if bool(item.get("sign_flip", False)) else 1.0
            mujoco_qacc = sign * float(mjcf.data.qacc[qvel_idx])
            err = abs(opensim_qacc - mujoco_qacc)
            status = "evaluated"
        except Exception as exc:
            opensim_qacc = mujoco_qacc = err = np.nan
            status = f"skipped: {exc}"
        rows.append(
            {
                "experiment": "passive_no_contact_instant_acceleration",
                "opensim_coordinate": oname,
                "mujoco_qvel": mname,
                "coordinate_role": role,
                "opensim_qacc": opensim_qacc,
                "mujoco_qacc": mujoco_qacc,
                "absolute_error": err,
                "relative_error": rel_error(err, max(abs(opensim_qacc), abs(mujoco_qacc))),
                "status": status,
            }
        )
    return rows


def _status(df: Any, mapping: MappingConfig) -> str:
    if df.empty:
        return "not evaluated"
    finite = df[(df["status"] == "evaluated") & (df["coordinate_role"] == "independent")]
    if finite.empty:
        return "not evaluated"
    abs_err = np.array(finite["absolute_error"], dtype=float)
    rel_err = np.array(finite["relative_error"], dtype=float)
    reference = np.maximum(np.abs(np.array(finite["opensim_qacc"], dtype=float)), np.abs(np.array(finite["mujoco_qacc"], dtype=float)))
    warn_abs = float(mapping.thresholds.get("forward_dynamics_accel_warning", 5.0))
    fail_abs = float(mapping.thresholds.get("forward_dynamics_accel_fail", 20.0))
    warn_rel = float(mapping.thresholds.get("forward_dynamics_accel_warning_rel", 0.05))
    fail_rel = float(mapping.thresholds.get("forward_dynamics_accel_fail_rel", 0.10))
    if np.nanmax(abs_err) > fail_abs or np.nanmax(rel_err[np.isfinite(rel_err)]) > fail_rel:
        return "failed"
    if np.nanmax(abs_err) > warn_abs or np.nanmax(rel_err[np.isfinite(rel_err)]) > warn_rel:
        return "warning"
    return "passed"


def run(osim: Any, mjcf: Any, mapping: MappingConfig, out_dir: Path) -> dict[str, Any]:
    planned = _plan_rows(mapping)
    if planned:
        write_csv(out_dir / "forward_dynamics_plan.csv", planned)
    rows = _instant_acceleration_rows(osim, mjcf, mapping)
    df = write_csv(out_dir / "forward_dynamics_smoke_test.csv", rows)
    write_markdown(
        out_dir / "forward_dynamics_notes.md",
        "# Forward Dynamics Notes\n\n"
        "The executable gate compares t=0 generalized accelerations at the matched neutral pose with zero speeds and zero controls.\n"
        "MuJoCo contact is disabled for this no-contact smoke test. This avoids contact discontinuities and catches force, sign, inertia, "
        "coordinate-adapter and passive-actuation mismatches before long rollouts amplify them.\n"
        "The planned 50 ms torque-pulse, 200 ms matched-activation and contact-drop rollouts should only be trusted after this "
        "instant-acceleration gate passes.\n",
    )
    files = ["forward_dynamics_smoke_test.csv", "forward_dynamics_notes.md"]
    worst = write_worst_csv(out_dir / "diagnostics" / "forward_dynamics_worst_acceleration_error.csv", df, "absolute_error")
    if worst:
        files.append(f"diagnostics/{worst}")
    if planned:
        files.append("forward_dynamics_plan.csv")
    finite = df[(df["status"] == "evaluated") & (df["coordinate_role"] == "independent")] if not df.empty else df
    status = _status(df, mapping)
    reason = None
    if status == "failed":
        reason = "Neutral no-contact forward dynamics acceleration mismatch exceeds thresholds; inspect diagnostics/forward_dynamics_worst_acceleration_error.csv."
    elif status == "not evaluated":
        reason = "No directly comparable independent-coordinate acceleration rows."
    return {
        "status": status,
        **({"reason": reason} if reason else {}),
        "planned_experiments": len(planned),
        "evaluated_rows": int(len(finite)) if not finite.empty else 0,
        "max_abs_error": float(finite["absolute_error"].max()) if not finite.empty else None,
        "rmse_abs_error": rmse(finite["absolute_error"]) if not finite.empty else None,
        "warning_threshold": float(mapping.thresholds.get("forward_dynamics_accel_warning", 5.0)),
        "failure_threshold": float(mapping.thresholds.get("forward_dynamics_accel_fail", 20.0)),
        "files": files,
    }
