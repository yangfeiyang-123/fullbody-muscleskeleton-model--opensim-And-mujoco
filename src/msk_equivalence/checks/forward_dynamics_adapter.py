from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from msk_equivalence.adapters.runtime_dynamics import (
    coordinate_group,
    fit_residual_adapter,
    qacc_status,
    row_relative_error,
    split_indices,
)
from msk_equivalence.checks.forward_dynamics import (
    _mass_adapter_random_qacc_rows,
    _random_activation_vectors,
    _sensitivity_audit_rows,
)
from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import rmse, write_csv, write_markdown, write_worst_csv


def _finite_rows(df: Any) -> Any:
    return df[(df["status"] == "evaluated") & (df["coordinate_role"] == "independent")] if not df.empty else df


def _split_for_vector(vector_index: int, splits: dict[str, list[int]]) -> str:
    for name, indices in splits.items():
        if vector_index in indices:
            return name
    return "unused"


def _residual_vectors(df: Any) -> tuple[list[str], dict[int, np.ndarray]]:
    finite = _finite_rows(df)
    if finite.empty:
        return [], {}
    coordinate_names = [str(name) for name in finite[finite["vector_index"] == finite["vector_index"].min()]["opensim_coordinate"]]
    residuals: dict[int, np.ndarray] = {}
    for vector_idx, group in finite.groupby("vector_index", sort=True):
        ordered = group.reset_index(drop=True)
        residuals[int(vector_idx)] = np.array(ordered["mujoco_qacc"], dtype=float) - np.array(ordered["opensim_mass_adapter_qacc"], dtype=float)
    return coordinate_names, residuals


def _corrected_rows(df: Any, vectors: list[dict[str, float]], splits: dict[str, list[int]], adapter: Any) -> list[dict[str, Any]]:
    finite = _finite_rows(df)
    if finite.empty or adapter is None:
        return []
    rows: list[dict[str, Any]] = []
    for vector_idx, group in finite.groupby("vector_index", sort=True):
        vector_index = int(vector_idx)
        predicted_residual = adapter.predict(vectors[vector_index])
        split = _split_for_vector(vector_index, splits)
        ordered = group.reset_index(drop=True)
        for row_idx, row in ordered.iterrows():
            mass_qacc = float(row["opensim_mass_adapter_qacc"])
            corrected = mass_qacc + float(predicted_residual[row_idx])
            target = float(row["mujoco_qacc"])
            err = abs(corrected - target)
            oname = str(row["opensim_coordinate"])
            rows.append(
                {
                    "experiment": "runtime_adapter_sparse_random_activation_qacc",
                    "split": split,
                    "vector_index": vector_index,
                    "opensim_coordinate": oname,
                    "mujoco_qvel": row["mujoco_qvel"],
                    "coordinate_group": coordinate_group(oname),
                    "coordinate_role": row["coordinate_role"],
                    "opensim_raw_no_correction_qacc": float(row["opensim_raw_no_correction_qacc"]),
                    "opensim_mass_adapter_qacc": mass_qacc,
                    "adapter_residual_qacc": float(predicted_residual[row_idx]),
                    "opensim_runtime_adapter_qacc": corrected,
                    "mujoco_qacc": target,
                    "absolute_error": err,
                    "relative_error": row_relative_error(corrected, target),
                    "active_muscle_count": int(row.get("active_muscle_count", 0)),
                    "status": "evaluated",
                }
            )
    return rows


def _group_summary_rows(df: Any) -> list[dict[str, Any]]:
    finite = _finite_rows(df)
    if finite.empty:
        return []
    rows: list[dict[str, Any]] = []
    for (split, group_name), group in finite.groupby(["split", "coordinate_group"], sort=True):
        rows.append(
            {
                "split": split,
                "coordinate_group": group_name,
                "evaluated_rows": int(len(group)),
                "max_abs_error": float(group["absolute_error"].max()),
                "rmse_abs_error": rmse(group["absolute_error"]),
                "max_relative_error": float(group["relative_error"].max()),
                "worst_coordinate": str(group.sort_values("absolute_error", ascending=False).iloc[0]["opensim_coordinate"]),
                "status": "evaluated",
            }
        )
    return rows


def run(osim: Any, mjcf: Any, mapping: MappingConfig, out_dir: Path) -> dict[str, Any]:
    sensitivity_rows, sensitivity_metrics, _, opensim_sensitivity, mujoco_with_armature_sensitivity, probe_context = _sensitivity_audit_rows(osim, mjcf, mapping)
    sensitivity_df = write_csv(out_dir / "diagnostics" / "forward_dynamics_adapter_sensitivity_audit.csv", sensitivity_rows)
    mass_rows = _mass_adapter_random_qacc_rows(mjcf, mapping, opensim_sensitivity, mujoco_with_armature_sensitivity, probe_context)
    mass_df = write_csv(out_dir / "diagnostics" / "forward_dynamics_adapter_mass_stage_qacc.csv", mass_rows)

    vectors = _random_activation_vectors(mapping)
    splits = split_indices(len(vectors), mapping)
    coordinate_names, residuals = _residual_vectors(mass_df)
    adapter = fit_residual_adapter(vectors, coordinate_names, residuals, splits["train"], mapping)
    if adapter is not None:
        adapter.save_npz(out_dir / "adapter" / "runtime_dynamics_residual_linear.npz")
    rows = _corrected_rows(mass_df, vectors, splits, adapter)
    df = write_csv(out_dir / "forward_dynamics_adapter_qacc.csv", rows)
    group_df = write_csv(out_dir / "diagnostics" / "forward_dynamics_adapter_group_summary.csv", _group_summary_rows(df))

    worst = write_worst_csv(out_dir / "diagnostics" / "forward_dynamics_adapter_worst_qacc_error.csv", df, "absolute_error")
    mass_worst = write_worst_csv(out_dir / "diagnostics" / "forward_dynamics_adapter_mass_stage_worst_qacc_error.csv", mass_df, "absolute_error")
    group_worst = write_worst_csv(out_dir / "diagnostics" / "forward_dynamics_adapter_worst_group_error.csv", group_df, "max_abs_error")
    sensitivity_worst = write_worst_csv(out_dir / "diagnostics" / "forward_dynamics_adapter_sensitivity_worst_error.csv", sensitivity_df, "armature_adjusted_diag_abs_error")

    finite = _finite_rows(df)
    test_finite = finite[finite["split"] == "test"] if not finite.empty else finite
    status = qacc_status(df, mapping, split="test")
    if status == "not evaluated" and adapter is None:
        reason = "Runtime adapter could not be fitted because mass/armature sensitivity diagnostics were unavailable."
    elif status == "not evaluated":
        reason = "No held-out test rows were available for runtime adapter qacc validation."
    elif status == "failed":
        reason = "Held-out runtime-adapter qacc mismatch exceeds thresholds; inspect diagnostics/forward_dynamics_adapter_worst_qacc_error.csv."
    else:
        reason = None

    files = [
        "forward_dynamics_adapter_qacc.csv",
        "forward_dynamics_adapter_report.md",
        "diagnostics/forward_dynamics_adapter_sensitivity_audit.csv",
        "diagnostics/forward_dynamics_adapter_mass_stage_qacc.csv",
        "diagnostics/forward_dynamics_adapter_group_summary.csv",
    ]
    if adapter is not None:
        files.append("adapter/runtime_dynamics_residual_linear.npz")
    for item in (worst, mass_worst, group_worst, sensitivity_worst):
        if item:
            files.append(f"diagnostics/{item}")

    write_markdown(
        out_dir / "forward_dynamics_adapter_report.md",
        "\n".join(
            [
                "# Forward Dynamics Runtime Adapter Gate",
                "",
                "This gate evaluates an explicit runtime dynamics adapter. It does not modify the OpenSim `.osim` file and is not a native OpenSim Level 4 pass.",
                "",
                f"- status: `{status}`",
                f"- split_indices: `{splits}`",
                f"- train_vectors: `{len(splits['train'])}`",
                f"- val_vectors: `{len(splits['val'])}`",
                f"- test_vectors: `{len(splits['test'])}`",
                f"- evaluated_test_rows: `{int(len(test_finite)) if not test_finite.empty else 0}`",
                f"- test_max_abs_error: `{float(test_finite['absolute_error'].max()) if not test_finite.empty else None}`",
                f"- test_rmse_abs_error: `{rmse(test_finite['absolute_error']) if not test_finite.empty else None}`",
                f"- test_max_relative_error: `{float(test_finite['relative_error'].max()) if not test_finite.empty else None}`",
                f"- sensitivity_no_armature_fro_relative_error: `{sensitivity_metrics.get('sensitivity_no_armature_fro_relative_error')}`",
                f"- sensitivity_armature_adjusted_fro_relative_error: `{sensitivity_metrics.get('sensitivity_armature_adjusted_fro_relative_error')}`",
                "",
                "The adapter is fitted only on the configured train split and gated on the held-out test split.",
                "Group summaries are emitted so distal hand/thumb/wrist failures cannot be hidden by full-body averages.",
            ]
        )
        + "\n",
    )

    return {
        "status": status,
        **({"reason": reason} if reason else {}),
        "train_vectors": len(splits["train"]),
        "val_vectors": len(splits["val"]),
        "test_vectors": len(splits["test"]),
        "evaluated_rows": int(len(finite)) if not finite.empty else 0,
        "evaluated_test_rows": int(len(test_finite)) if not test_finite.empty else 0,
        "test_max_abs_error": float(test_finite["absolute_error"].max()) if not test_finite.empty else None,
        "test_rmse_abs_error": rmse(test_finite["absolute_error"]) if not test_finite.empty else None,
        "test_max_relative_error": float(test_finite["relative_error"].max()) if not test_finite.empty else None,
        "sensitivity_no_armature_fro_relative_error": sensitivity_metrics.get("sensitivity_no_armature_fro_relative_error"),
        "sensitivity_armature_adjusted_fro_relative_error": sensitivity_metrics.get("sensitivity_armature_adjusted_fro_relative_error"),
        "note": "Explicit runtime adapter gate. Passing this check supports the OpenSim+adapter route only, not pure native `.osim` Level 4 equivalence.",
        "files": files,
    }
