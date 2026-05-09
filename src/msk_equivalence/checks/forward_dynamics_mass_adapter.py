from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

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
from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import rmse, write_csv, write_markdown, write_worst_csv


def _status(rows: Any, mapping: MappingConfig) -> str:
    if rows.empty:
        return "not evaluated"
    finite = rows[(rows["status"] == "evaluated") & (rows["coordinate_role"] == "independent")]
    if finite.empty:
        return "not evaluated"
    abs_err = np.array(finite["absolute_error"], dtype=float)
    rel_err = np.array(finite["relative_error"], dtype=float)
    warn_abs = float(mapping.thresholds.get("forward_dynamics_accel_warning", 5.0))
    fail_abs = float(mapping.thresholds.get("forward_dynamics_accel_fail", 20.0))
    warn_rel = float(mapping.thresholds.get("forward_dynamics_accel_warning_rel", 0.05))
    fail_rel = float(mapping.thresholds.get("forward_dynamics_accel_fail_rel", 0.10))
    finite_rel = rel_err[np.isfinite(rel_err)]
    max_rel = float(np.max(finite_rel)) if finite_rel.size else 0.0
    if float(np.nanmax(abs_err)) > fail_abs or max_rel > fail_rel:
        return "failed"
    if float(np.nanmax(abs_err)) > warn_abs or max_rel > warn_rel:
        return "warning"
    return "passed"


def _vector_indices(mapping: MappingConfig) -> tuple[list[int], list[dict[str, float]]]:
    vectors = _random_activation_vectors(mapping)
    debug = mapping.raw.get("_debug", {})
    debug_vector_index = debug.get("vector_index") if isinstance(debug, dict) else None
    if debug_vector_index is None:
        return list(range(len(vectors))), vectors
    idx = int(debug_vector_index)
    if 0 <= idx < len(vectors):
        return [idx], [vectors[idx]]
    return [], []


def _residual_force_rows(rows: Any, mapping: MappingConfig, mujoco_with_armature_sensitivity: np.ndarray | None) -> list[dict[str, Any]]:
    if rows.empty or mujoco_with_armature_sensitivity is None:
        return []
    finite = rows[(rows["status"] == "evaluated") & (rows["coordinate_role"] == "independent")].copy()
    if finite.empty:
        return []
    try:
        sensitivity_inverse = np.linalg.pinv(mujoco_with_armature_sensitivity, rcond=1e-12)
    except Exception:
        return []
    vector_indices, vectors = _vector_indices(mapping)
    active_by_vector = {idx: " ".join(sorted(vector)) for idx, vector in zip(vector_indices, vectors)}
    out: list[dict[str, Any]] = []
    for vector_idx, group in finite.groupby("vector_index", sort=False):
        ordered = group.reset_index(drop=True)
        error_qacc = np.array(ordered["mujoco_qacc"], dtype=float) - np.array(ordered["opensim_mass_adapter_qacc"], dtype=float)
        residual_force = sensitivity_inverse @ error_qacc
        for row_idx, row in ordered.iterrows():
            value = float(residual_force[row_idx])
            out.append(
                {
                    "experiment": "diagnostic_mass_adapter_residual_generalized_force",
                    "vector_index": int(vector_idx),
                    "opensim_coordinate": row["opensim_coordinate"],
                    "mujoco_qvel": row["mujoco_qvel"],
                    "opensim_mass_adapter_qacc": float(row["opensim_mass_adapter_qacc"]),
                    "mujoco_qacc": float(row["mujoco_qacc"]),
                    "qacc_residual": float(error_qacc[row_idx]),
                    "estimated_required_generalized_force": value,
                    "absolute_required_generalized_force": abs(value),
                    "active_mujoco_actuators": active_by_vector.get(int(vector_idx), ""),
                    "diagnostic_only": True,
                    "note": "Estimated by applying the inverse MuJoCo with-armature sensitivity to the mass-adapter qacc residual.",
                    "status": "evaluated",
                }
            )
    return out


def _zero_activation_mass_adapter_rows(
    mjcf: Any,
    mapping: MappingConfig,
    opensim_sensitivity: np.ndarray | None,
    mujoco_with_armature_sensitivity: np.ndarray | None,
    context: tuple[Any, dict[str, Any], dict[str, float], dict[str, float], list[dict[str, Any]]] | None,
) -> list[dict[str, Any]]:
    if opensim_sensitivity is None or mujoco_with_armature_sensitivity is None or context is None:
        return []
    probe_osim, probes, opensim_pose, mujoco_pose, coordinate_items = context
    _set_probe_expressions(probes)
    probe_osim.state = probe_osim.model.initSystem()
    try:
        dynamics_adapter = mujoco_with_armature_sensitivity @ np.linalg.pinv(opensim_sensitivity, rcond=1e-12)
    except Exception:
        return []

    probe_osim.set_pose(opensim_pose)
    _set_opensim_speeds(probe_osim, {})
    _set_opensim_activations(probe_osim, {})
    try:
        probe_osim.model.realizeAcceleration(probe_osim.state)
        opensim_raw = _opensim_qacc_vector(probe_osim, coordinate_items, mapping)
    except Exception as exc:
        return [{"experiment": "diagnostic_mass_adapter_zero_activation_qacc", "vector_index": -1, "status": f"skipped: {exc}"}]

    _mujoco_forward(mjcf, mujoco_pose, {}, {}, disable_contact=True)
    mujoco_qacc = _mujoco_qacc_vector(mjcf, coordinate_items, mapping)
    opensim_adapter_qacc = dynamics_adapter @ opensim_raw
    rows: list[dict[str, Any]] = []
    for row_idx, item in enumerate(coordinate_items):
        oname = mapping.side_name(item, "opensim")
        mname = mapping.side_name(item, "mujoco")
        err = abs(float(opensim_adapter_qacc[row_idx]) - float(mujoco_qacc[row_idx]))
        rows.append(
            {
                "experiment": "diagnostic_mass_adapter_zero_activation_qacc",
                "opensim_coordinate": oname,
                "mujoco_qvel": mname,
                "coordinate_role": "independent",
                "opensim_mass_adapter_qacc": float(opensim_adapter_qacc[row_idx]),
                "opensim_raw_no_correction_qacc": float(opensim_raw[row_idx]),
                "mujoco_qacc": float(mujoco_qacc[row_idx]),
                "absolute_error": err,
                "relative_error": err / max(abs(float(opensim_adapter_qacc[row_idx])), abs(float(mujoco_qacc[row_idx])), 1e-12),
                "status": "evaluated",
                "vector_index": -1,
                "active_muscle_count": 0,
                "diagnostic_only": True,
                "note": "Diagnostic only: zero-activation no-contact qacc through the same mass adapter. Correction-force probes are disabled.",
            }
        )
    return rows


def _zero_baseline_force_vector(
    zero_residual_force_rows: Any,
    coordinate_items: list[dict[str, Any]],
    mapping: MappingConfig,
) -> np.ndarray | None:
    if zero_residual_force_rows.empty:
        return None
    finite = zero_residual_force_rows[zero_residual_force_rows["status"] == "evaluated"].copy()
    if finite.empty:
        return None
    force_by_coordinate = {
        str(row["opensim_coordinate"]): float(row["estimated_required_generalized_force"])
        for _, row in finite.iterrows()
    }
    values: list[float] = []
    for item in coordinate_items:
        oname = mapping.side_name(item, "opensim")
        if not oname or oname not in force_by_coordinate:
            return None
        values.append(force_by_coordinate[oname])
    return np.array(values, dtype=float)


def _zero_baseline_corrected_rows(
    rows: Any,
    zero_residual_force_rows: Any,
    mapping: MappingConfig,
    mujoco_with_armature_sensitivity: np.ndarray | None,
    context: tuple[Any, dict[str, Any], dict[str, float], dict[str, float], list[dict[str, Any]]] | None,
) -> list[dict[str, Any]]:
    if rows.empty or mujoco_with_armature_sensitivity is None or context is None:
        return []
    _, _, _, _, coordinate_items = context
    zero_force = _zero_baseline_force_vector(zero_residual_force_rows, coordinate_items, mapping)
    if zero_force is None:
        return []
    baseline_qacc = mujoco_with_armature_sensitivity @ zero_force
    finite = rows[(rows["status"] == "evaluated") & (rows["coordinate_role"] == "independent")].copy()
    if finite.empty:
        return []
    out: list[dict[str, Any]] = []
    for vector_idx, group in finite.groupby("vector_index", sort=False):
        ordered = group.reset_index(drop=True)
        for row_idx, row in ordered.iterrows():
            corrected = float(row["opensim_mass_adapter_qacc"]) + float(baseline_qacc[row_idx])
            mujoco_qacc = float(row["mujoco_qacc"])
            err = abs(corrected - mujoco_qacc)
            out.append(
                {
                    "experiment": "diagnostic_zero_baseline_corrected_mass_adapter_qacc",
                    "vector_index": int(vector_idx),
                    "opensim_coordinate": row["opensim_coordinate"],
                    "mujoco_qvel": row["mujoco_qvel"],
                    "coordinate_role": row["coordinate_role"],
                    "opensim_mass_adapter_qacc": float(row["opensim_mass_adapter_qacc"]),
                    "zero_baseline_qacc_correction": float(baseline_qacc[row_idx]),
                    "opensim_zero_baseline_corrected_qacc": corrected,
                    "mujoco_qacc": mujoco_qacc,
                    "absolute_error": err,
                    "relative_error": err / max(abs(corrected), abs(mujoco_qacc), 1e-12),
                    "status": "evaluated",
                    "active_muscle_count": int(row.get("active_muscle_count", 0)),
                    "diagnostic_only": True,
                    "note": "Diagnostic only: applies the zero-activation residual generalized-force baseline after the mass adapter.",
                }
            )
    return out


def run(osim: Any, mjcf: Any, mapping: MappingConfig, out_dir: Path) -> dict[str, Any]:
    sensitivity_rows, sensitivity_metrics, _, opensim_sensitivity, mujoco_with_armature_sensitivity, probe_context = _sensitivity_audit_rows(osim, mjcf, mapping)
    sensitivity_df = write_csv(out_dir / "diagnostics" / "forward_dynamics_mass_adapter_sensitivity_audit.csv", sensitivity_rows)
    zero_rows = _zero_activation_mass_adapter_rows(mjcf, mapping, opensim_sensitivity, mujoco_with_armature_sensitivity, probe_context)
    zero_df = write_csv(out_dir / "diagnostics" / "forward_dynamics_mass_adapter_zero_activation_qacc.csv", zero_rows)
    zero_residual_force_rows = _residual_force_rows(zero_df, mapping, mujoco_with_armature_sensitivity)
    zero_residual_force_df = write_csv(out_dir / "diagnostics" / "forward_dynamics_mass_adapter_zero_activation_residual_generalized_force.csv", zero_residual_force_rows)
    rows = _mass_adapter_random_qacc_rows(mjcf, mapping, opensim_sensitivity, mujoco_with_armature_sensitivity, probe_context)
    df = write_csv(out_dir / "forward_dynamics_mass_adapter_random_activation_qacc.csv", rows)
    residual_force_rows = _residual_force_rows(df, mapping, mujoco_with_armature_sensitivity)
    residual_force_df = write_csv(out_dir / "diagnostics" / "forward_dynamics_mass_adapter_residual_generalized_force.csv", residual_force_rows)
    zero_baseline_rows = _zero_baseline_corrected_rows(
        df,
        zero_residual_force_df,
        mapping,
        mujoco_with_armature_sensitivity,
        probe_context,
    )
    zero_baseline_df = write_csv(out_dir / "diagnostics" / "forward_dynamics_mass_adapter_zero_baseline_corrected_qacc.csv", zero_baseline_rows)
    zero_worst = write_worst_csv(out_dir / "diagnostics" / "forward_dynamics_mass_adapter_zero_activation_worst_qacc_error.csv", zero_df, "absolute_error")
    zero_residual_worst = write_worst_csv(
        out_dir / "diagnostics" / "forward_dynamics_mass_adapter_zero_activation_residual_generalized_force_worst.csv",
        zero_residual_force_df,
        "absolute_required_generalized_force",
    )
    worst = write_worst_csv(out_dir / "diagnostics" / "forward_dynamics_mass_adapter_random_activation_worst_qacc_error.csv", df, "absolute_error")
    zero_baseline_worst = write_worst_csv(
        out_dir / "diagnostics" / "forward_dynamics_mass_adapter_zero_baseline_corrected_worst_qacc_error.csv",
        zero_baseline_df,
        "absolute_error",
    )
    residual_worst = write_worst_csv(
        out_dir / "diagnostics" / "forward_dynamics_mass_adapter_residual_generalized_force_worst.csv",
        residual_force_df,
        "absolute_required_generalized_force",
    )
    sensitivity_worst = write_worst_csv(out_dir / "diagnostics" / "forward_dynamics_mass_adapter_sensitivity_worst_error.csv", sensitivity_df, "armature_adjusted_diag_abs_error")
    files = [
        "forward_dynamics_mass_adapter_random_activation_qacc.csv",
        "forward_dynamics_mass_adapter_report.md",
        "diagnostics/forward_dynamics_mass_adapter_sensitivity_audit.csv",
        "diagnostics/forward_dynamics_mass_adapter_zero_activation_qacc.csv",
        "diagnostics/forward_dynamics_mass_adapter_zero_activation_residual_generalized_force.csv",
        "diagnostics/forward_dynamics_mass_adapter_residual_generalized_force.csv",
        "diagnostics/forward_dynamics_mass_adapter_zero_baseline_corrected_qacc.csv",
    ]
    if zero_worst:
        files.append(f"diagnostics/{zero_worst}")
    if zero_residual_worst:
        files.append(f"diagnostics/{zero_residual_worst}")
    if worst:
        files.append(f"diagnostics/{worst}")
    if zero_baseline_worst:
        files.append(f"diagnostics/{zero_baseline_worst}")
    if residual_worst:
        files.append(f"diagnostics/{residual_worst}")
    if sensitivity_worst:
        files.append(f"diagnostics/{sensitivity_worst}")
    finite = df[(df["status"] == "evaluated") & (df["coordinate_role"] == "independent")] if not df.empty else df
    zero_finite = zero_df[(zero_df["status"] == "evaluated") & (zero_df["coordinate_role"] == "independent")] if not zero_df.empty else zero_df
    residual_finite = residual_force_df[residual_force_df["status"] == "evaluated"] if not residual_force_df.empty else residual_force_df
    zero_residual_finite = (
        zero_residual_force_df[zero_residual_force_df["status"] == "evaluated"] if not zero_residual_force_df.empty else zero_residual_force_df
    )
    zero_baseline_finite = (
        zero_baseline_df[(zero_baseline_df["status"] == "evaluated") & (zero_baseline_df["coordinate_role"] == "independent")]
        if not zero_baseline_df.empty
        else zero_baseline_df
    )
    max_abs = float(finite["absolute_error"].max()) if not finite.empty else None
    rmse_abs = rmse(finite["absolute_error"]) if not finite.empty else None
    max_rel = float(finite["relative_error"].max()) if not finite.empty else None
    zero_max_abs = float(zero_finite["absolute_error"].max()) if not zero_finite.empty else None
    zero_rmse_abs = rmse(zero_finite["absolute_error"]) if not zero_finite.empty else None
    max_force = float(residual_finite["absolute_required_generalized_force"].max()) if not residual_finite.empty else None
    rmse_force = rmse(residual_finite["estimated_required_generalized_force"]) if not residual_finite.empty else None
    zero_max_force = (
        float(zero_residual_finite["absolute_required_generalized_force"].max()) if not zero_residual_finite.empty else None
    )
    zero_rmse_force = rmse(zero_residual_finite["estimated_required_generalized_force"]) if not zero_residual_finite.empty else None
    zero_baseline_max_abs = float(zero_baseline_finite["absolute_error"].max()) if not zero_baseline_finite.empty else None
    zero_baseline_rmse_abs = rmse(zero_baseline_finite["absolute_error"]) if not zero_baseline_finite.empty else None
    zero_baseline_max_rel = float(zero_baseline_finite["relative_error"].max()) if not zero_baseline_finite.empty else None
    status = _status(df, mapping)
    write_markdown(
        out_dir / "forward_dynamics_mass_adapter_report.md",
        "\n".join(
            [
                "# Forward Dynamics Mass-Adapter Diagnostic",
                "",
                "This diagnostic maps OpenSim raw qacc through an explicit MuJoCo with-armature generalized-force sensitivity adapter.",
                "It is not a native `.osim` behavior gate and must not be used to hide the strict `forward_dynamics` failure.",
                "",
                f"- status: `{status}`",
                f"- max_abs_error: `{max_abs}`",
                f"- rmse_abs_error: `{rmse_abs}`",
                f"- max_relative_error: `{max_rel}`",
                f"- zero_activation_max_abs_error: `{zero_max_abs}`",
                f"- zero_activation_rmse_abs_error: `{zero_rmse_abs}`",
                f"- residual_generalized_force_max_abs: `{max_force}`",
                f"- residual_generalized_force_rmse: `{rmse_force}`",
                f"- zero_activation_residual_generalized_force_max_abs: `{zero_max_force}`",
                f"- zero_activation_residual_generalized_force_rmse: `{zero_rmse_force}`",
                f"- zero_baseline_corrected_random_activation_max_abs_error: `{zero_baseline_max_abs}`",
                f"- zero_baseline_corrected_random_activation_rmse_abs_error: `{zero_baseline_rmse_abs}`",
                f"- zero_baseline_corrected_random_activation_max_relative_error: `{zero_baseline_max_rel}`",
                f"- sensitivity_no_armature_fro_relative_error: `{sensitivity_metrics.get('sensitivity_no_armature_fro_relative_error')}`",
                f"- sensitivity_armature_adjusted_fro_relative_error: `{sensitivity_metrics.get('sensitivity_armature_adjusted_fro_relative_error')}`",
            ]
        )
        + "\n",
    )
    return {
        "status": status,
        "evaluated_rows": int(len(finite)) if not finite.empty else 0,
        "max_abs_error": max_abs,
        "rmse_abs_error": rmse_abs,
        "max_relative_error": max_rel,
        "zero_activation_max_abs_error": zero_max_abs,
        "zero_activation_rmse_abs_error": zero_rmse_abs,
        "residual_generalized_force_max_abs": max_force,
        "residual_generalized_force_rmse": rmse_force,
        "zero_activation_residual_generalized_force_max_abs": zero_max_force,
        "zero_activation_residual_generalized_force_rmse": zero_rmse_force,
        "zero_baseline_corrected_random_activation_max_abs_error": zero_baseline_max_abs,
        "zero_baseline_corrected_random_activation_rmse_abs_error": zero_baseline_rmse_abs,
        "zero_baseline_corrected_random_activation_max_relative_error": zero_baseline_max_rel,
        "sensitivity_no_armature_fro_relative_error": sensitivity_metrics.get("sensitivity_no_armature_fro_relative_error"),
        "sensitivity_armature_adjusted_fro_relative_error": sensitivity_metrics.get("sensitivity_armature_adjusted_fro_relative_error"),
        "note": "Diagnostic only. Passing this check would support an external runtime dynamics-adapter route; it is not native OpenSim Level 4 equivalence.",
        "files": files,
    }
