from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from msk_equivalence.adapters.runtime_dynamics import fit_residual_adapter, rollout_status, row_relative_error, split_indices
from msk_equivalence.checks.forward_dynamics import (
    _mass_adapter_random_qacc_rows,
    _mujoco_forward,
    _mujoco_qacc_vector,
    _opensim_qacc_vector,
    _random_activation_vectors,
    _sensitivity_audit_rows,
    _set_opensim_activations,
    _set_opensim_speeds,
    _set_probe_expressions,
)
from msk_equivalence.checks.kinematics import _pose_values
from msk_equivalence.checks.inverse_dynamics import _coupler_constraints, _with_coupled_coordinates
from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import rmse, write_csv, write_markdown, write_worst_csv


def _finite_rows(df: Any) -> Any:
    return df[(df["status"] == "evaluated") & (df["coordinate_role"] == "independent")] if not df.empty else df


def _residual_vectors(df: Any) -> tuple[list[str], dict[int, np.ndarray]]:
    finite = _finite_rows(df)
    if finite.empty:
        return [], {}
    coordinate_names = [str(name) for name in finite[finite["vector_index"] == finite["vector_index"].min()]["opensim_coordinate"]]
    residuals: dict[int, np.ndarray] = {}
    for vector_idx, group in finite.groupby("vector_index", sort=True):
        residuals[int(vector_idx)] = np.array(group.reset_index(drop=True)["mujoco_qacc"], dtype=float) - np.array(
            group.reset_index(drop=True)["opensim_mass_adapter_qacc"], dtype=float
        )
    return coordinate_names, residuals


def _rollout_settings(mapping: MappingConfig) -> tuple[float, float]:
    cfg = mapping.raw.get("_adapter_config", {})
    rollout = cfg.get("rollout", {}) if isinstance(cfg, dict) and isinstance(cfg.get("rollout", {}), dict) else {}
    return float(rollout.get("horizon_s", 0.05)), float(rollout.get("dt_s", 0.002))


def _state_error_rows(
    step: int,
    time_s: float,
    q_osim: dict[str, float],
    qdot_osim: dict[str, float],
    q_mujoco: dict[str, float],
    qdot_mujoco: dict[str, float],
    coordinate_items: list[dict[str, Any]],
    mapping: MappingConfig,
) -> list[dict[str, Any]]:
    rows = []
    for item in coordinate_items:
        oname = mapping.side_name(item, "opensim")
        mname = mapping.side_name(item, "mujoco")
        if not oname or not mname:
            continue
        sign = -1.0 if bool(item.get("sign_flip", False)) else 1.0
        oq = float(q_osim.get(oname, 0.0))
        mq = sign * float(q_mujoco.get(mname, 0.0))
        ov = float(qdot_osim.get(oname, 0.0))
        mv = sign * float(qdot_mujoco.get(mname, 0.0))
        rows.append(
            {
                "experiment": "runtime_adapter_no_contact_rollout",
                "step": step,
                "time_s": time_s,
                "opensim_coordinate": oname,
                "mujoco_qvel": mname,
                "opensim_q": oq,
                "mujoco_q": mq,
                "q_error": abs(oq - mq),
                "opensim_qdot": ov,
                "mujoco_qdot": mv,
                "qdot_error": abs(ov - mv),
                "status": "evaluated",
            }
        )
    return rows


def run(osim: Any, mjcf: Any, mapping: MappingConfig, out_dir: Path) -> dict[str, Any]:
    _, _, _, opensim_sensitivity, mujoco_with_armature_sensitivity, context = _sensitivity_audit_rows(osim, mjcf, mapping)
    mass_rows = _mass_adapter_random_qacc_rows(mjcf, mapping, opensim_sensitivity, mujoco_with_armature_sensitivity, context)
    mass_df = write_csv(out_dir / "diagnostics" / "adapter_no_contact_rollout_mass_stage_qacc.csv", mass_rows)
    vectors = _random_activation_vectors(mapping)
    splits = split_indices(len(vectors), mapping)
    coordinate_names, residuals = _residual_vectors(mass_df)
    adapter = fit_residual_adapter(vectors, coordinate_names, residuals, splits["train"], mapping)
    if adapter is None or opensim_sensitivity is None or mujoco_with_armature_sensitivity is None or context is None:
        return {
            "status": "not evaluated",
            "reason": "Runtime adapter could not be fitted because sensitivity diagnostics were unavailable.",
            "files": ["diagnostics/adapter_no_contact_rollout_mass_stage_qacc.csv"],
        }
    try:
        dynamics_adapter = mujoco_with_armature_sensitivity @ np.linalg.pinv(opensim_sensitivity, rcond=1e-12)
    except Exception as exc:
        return {"status": "not evaluated", "reason": f"Could not build mass/armature adapter: {exc}", "files": []}

    vector_index = (splits["test"] or splits["val"] or splits["train"])[0]
    mujoco_activations = vectors[vector_index]
    predicted_residual = adapter.predict(mujoco_activations)
    probe_osim, probes, neutral_opensim_pose, neutral_mujoco_pose, coordinate_items = context
    _set_probe_expressions(probes)
    probe_osim.state = probe_osim.model.initSystem()

    sample = next((s for s in mapping.pose_samples if str(s.get("name", "")) == "neutral"), mapping.pose_samples[0])
    constraints = _coupler_constraints(probe_osim)
    base_opensim_pose = _with_coupled_coordinates(_pose_values(sample, mapping, "opensim"), constraints, mapping, "opensim")
    base_mujoco_pose = _with_coupled_coordinates(_pose_values(sample, mapping, "mujoco"), constraints, mapping, "mujoco")
    q_osim = {mapping.side_name(item, "opensim"): float(base_opensim_pose.get(mapping.side_name(item, "opensim") or "", 0.0)) for item in coordinate_items}
    q_mujoco = {mapping.side_name(item, "mujoco"): float(base_mujoco_pose.get(mapping.side_name(item, "mujoco") or "", 0.0)) for item in coordinate_items}
    qdot_osim = {mapping.side_name(item, "opensim"): 0.0 for item in coordinate_items}
    qdot_mujoco = {mapping.side_name(item, "mujoco"): 0.0 for item in coordinate_items}

    from msk_equivalence.checks.forward_dynamics import _build_activation_adapter, _opensim_activation_values

    activation_adapter = _build_activation_adapter(probe_osim, mjcf, mapping, neutral_opensim_pose, neutral_mujoco_pose, set(mujoco_activations))
    opensim_activations = _opensim_activation_values(mujoco_activations, mapping, activation_adapter)
    horizon, dt = _rollout_settings(mapping)
    steps = max(1, int(round(horizon / dt)))
    state_rows: list[dict[str, Any]] = []
    qacc_rows: list[dict[str, Any]] = []
    for step in range(steps + 1):
        time_s = step * dt
        opensim_pose = dict(base_opensim_pose)
        mujoco_pose = dict(base_mujoco_pose)
        opensim_pose.update({k: v for k, v in q_osim.items() if k})
        mujoco_pose.update({k: v for k, v in q_mujoco.items() if k})
        opensim_pose = _with_coupled_coordinates(opensim_pose, constraints, mapping, "opensim")
        mujoco_pose = _with_coupled_coordinates(mujoco_pose, constraints, mapping, "mujoco")

        probe_osim.set_pose(opensim_pose)
        _set_opensim_speeds(probe_osim, qdot_osim)
        _set_opensim_activations(probe_osim, opensim_activations)
        try:
            probe_osim.model.realizeAcceleration(probe_osim.state)
            opensim_raw = _opensim_qacc_vector(probe_osim, coordinate_items, mapping)
            adapter_qacc = dynamics_adapter @ opensim_raw + predicted_residual
        except Exception as exc:
            qacc_rows.append({"experiment": "runtime_adapter_no_contact_rollout_qacc", "step": step, "status": f"skipped: {exc}"})
            break
        _mujoco_forward(mjcf, mujoco_pose, qdot_mujoco, mujoco_activations, disable_contact=True)
        mujoco_qacc = _mujoco_qacc_vector(mjcf, coordinate_items, mapping)
        for row_idx, item in enumerate(coordinate_items):
            oname = mapping.side_name(item, "opensim")
            mname = mapping.side_name(item, "mujoco")
            pred = float(adapter_qacc[row_idx])
            target = float(mujoco_qacc[row_idx])
            qacc_rows.append(
                {
                    "experiment": "runtime_adapter_no_contact_rollout_qacc",
                    "vector_index": vector_index,
                    "step": step,
                    "time_s": time_s,
                    "opensim_coordinate": oname,
                    "mujoco_qvel": mname,
                    "coordinate_role": "independent",
                    "opensim_runtime_adapter_qacc": pred,
                    "mujoco_qacc": target,
                    "absolute_error": abs(pred - target),
                    "relative_error": row_relative_error(pred, target),
                    "status": "evaluated",
                }
            )
        state_rows.extend(_state_error_rows(step, time_s, q_osim, qdot_osim, q_mujoco, qdot_mujoco, coordinate_items, mapping))
        if step == steps:
            break
        for row_idx, item in enumerate(coordinate_items):
            oname = mapping.side_name(item, "opensim")
            mname = mapping.side_name(item, "mujoco")
            if not oname or not mname:
                continue
            sign = -1.0 if bool(item.get("sign_flip", False)) else 1.0
            qdot_osim[oname] = float(qdot_osim.get(oname, 0.0)) + float(adapter_qacc[row_idx]) * dt
            qdot_mujoco[mname] = float(qdot_mujoco.get(mname, 0.0)) + sign * float(mujoco_qacc[row_idx]) * dt
            q_osim[oname] = float(q_osim.get(oname, 0.0)) + qdot_osim[oname] * dt
            q_mujoco[mname] = float(q_mujoco.get(mname, 0.0)) + qdot_mujoco[mname] * dt

    state_df = write_csv(out_dir / "adapter_no_contact_rollout_state_error.csv", state_rows)
    qacc_df = write_csv(out_dir / "adapter_no_contact_rollout_qacc_error.csv", qacc_rows)
    q_worst = write_worst_csv(out_dir / "diagnostics" / "adapter_no_contact_rollout_worst_q_error.csv", state_df, "q_error")
    qdot_worst = write_worst_csv(out_dir / "diagnostics" / "adapter_no_contact_rollout_worst_qdot_error.csv", state_df, "qdot_error")
    qacc_worst = write_worst_csv(out_dir / "diagnostics" / "adapter_no_contact_rollout_worst_qacc_error.csv", qacc_df, "absolute_error")
    status = rollout_status(state_df, mapping)
    finite = state_df[state_df["status"] == "evaluated"] if not state_df.empty else state_df
    qacc_finite = _finite_rows(qacc_df)
    files = [
        "adapter_no_contact_rollout_state_error.csv",
        "adapter_no_contact_rollout_qacc_error.csv",
        "adapter_no_contact_rollout_report.md",
        "diagnostics/adapter_no_contact_rollout_mass_stage_qacc.csv",
    ]
    for item in (q_worst, qdot_worst, qacc_worst):
        if item:
            files.append(f"diagnostics/{item}")
    write_markdown(
        out_dir / "adapter_no_contact_rollout_report.md",
        "\n".join(
            [
                "# Runtime Adapter No-Contact Rollout Gate",
                "",
                "This gate rolls out OpenSim with the explicit runtime adapter and MuJoCo with contact disabled under the same fixed activation vector.",
                "",
                f"- status: `{status}`",
                f"- vector_index: `{vector_index}`",
                f"- horizon_s: `{horizon}`",
                f"- dt_s: `{dt}`",
                f"- max_q_error: `{float(finite['q_error'].max()) if not finite.empty else None}`",
                f"- max_qdot_error: `{float(finite['qdot_error'].max()) if not finite.empty else None}`",
                f"- qacc_max_abs_error: `{float(qacc_finite['absolute_error'].max()) if not qacc_finite.empty else None}`",
                f"- qacc_rmse_abs_error: `{rmse(qacc_finite['absolute_error']) if not qacc_finite.empty else None}`",
            ]
        )
        + "\n",
    )
    reason = "No-contact rollout drift exceeds thresholds." if status == "failed" else None
    return {
        "status": status,
        **({"reason": reason} if reason else {}),
        "vector_index": vector_index,
        "horizon_s": horizon,
        "dt_s": dt,
        "evaluated_rows": int(len(finite)) if not finite.empty else 0,
        "max_q_error": float(finite["q_error"].max()) if not finite.empty else None,
        "max_qdot_error": float(finite["qdot_error"].max()) if not finite.empty else None,
        "qacc_max_abs_error": float(qacc_finite["absolute_error"].max()) if not qacc_finite.empty else None,
        "qacc_rmse_abs_error": rmse(qacc_finite["absolute_error"]) if not qacc_finite.empty else None,
        "note": "Explicit runtime adapter no-contact rollout. This does not claim contact equivalence.",
        "files": files,
    }
