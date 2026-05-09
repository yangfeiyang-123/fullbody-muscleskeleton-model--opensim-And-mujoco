from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from msk_equivalence.checks.inverse_dynamics import _coupler_constraints, _with_coupled_coordinates
from msk_equivalence.checks.kinematics import _pose_values
from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import rmse, status_from_errors, write_csv, write_worst_csv


def _rotation_angle_error(lhs: np.ndarray, rhs: np.ndarray) -> float:
    relative = np.asarray(lhs, dtype=float).T @ np.asarray(rhs, dtype=float)
    cosine = float((np.trace(relative) - 1.0) / 2.0)
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def run(osim: Any, mjcf: Any, mapping: MappingConfig, out_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    constraints = _coupler_constraints(osim)
    for sample in mapping.pose_samples:
        sample_name = str(sample.get("name", "sample"))
        opensim_pose = _with_coupled_coordinates(_pose_values(sample, mapping, "opensim"), constraints, mapping, "opensim")
        mujoco_pose = _with_coupled_coordinates(_pose_values(sample, mapping, "mujoco"), constraints, mapping, "mujoco")
        osim.set_pose(opensim_pose)
        mjcf.set_pose(mujoco_pose)
        for item in mapping.bodies:
            oname = mapping.side_name(item, "opensim")
            mname = mapping.side_name(item, "mujoco")
            if not oname or not mname:
                continue
            try:
                error = _rotation_angle_error(osim.body_rotation_matrix(oname), mjcf.body_rotation_matrix(mname))
                status = "evaluated"
            except Exception as exc:
                error = np.nan
                status = f"skipped: {exc}"
            rows.append(
                {
                    "sample": sample_name,
                    "opensim_body": oname,
                    "mujoco_body": mname,
                    "orientation_error_rad": error,
                    "orientation_error_deg": float(np.degrees(error)) if np.isfinite(error) else np.nan,
                    "status": status,
                }
            )

    df = write_csv(out_dir / "orientation_audit_body_error.csv", rows)
    worst = write_worst_csv(out_dir / "diagnostics" / "orientation_audit_worst_body_error.csv", df, "orientation_error_rad")
    finite = [float(v) for v in df.get("orientation_error_rad", []) if np.isfinite(v)]
    max_error = float(np.max(finite)) if finite else None
    warn = float(mapping.thresholds.get("orientation_warning_rad", 1e-3))
    fail = float(mapping.thresholds.get("orientation_fail_rad", 1e-2))
    files = ["orientation_audit_body_error.csv"]
    if worst:
        files.append(f"diagnostics/{worst}")
    return {
        "status": status_from_errors(max_error, warn, fail),
        "evaluated_rows": len(finite),
        "max_orientation_error_rad": max_error,
        "max_orientation_error_deg": float(np.degrees(max_error)) if max_error is not None else None,
        "rmse_orientation_error_rad": rmse(finite),
        "warning_threshold_rad": warn,
        "failure_threshold_rad": fail,
        "note": "Compares mapped body ground-frame rotation matrices at configured pose samples. This closes the gap where position-only FK can pass while body frames are misaligned.",
        "files": files,
    }
