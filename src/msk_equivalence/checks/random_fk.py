from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from msk_equivalence.checks.forward_dynamics import _independent_coordinate_items
from msk_equivalence.checks.inverse_dynamics import _coupler_constraints, _with_coupled_coordinates
from msk_equivalence.checks.kinematics import _pose_values
from msk_equivalence.checks.orientation_audit import _rotation_angle_error
from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import norm_error, rmse, status_from_errors, write_csv, write_markdown, write_worst_csv


def _experiment(mapping: MappingConfig) -> dict[str, Any]:
    value = mapping.raw.get("random_fk", {})
    return value if isinstance(value, dict) else {}


def _coordinate_range(osim: Any, name: str) -> tuple[float, float] | None:
    try:
        coord = osim.model.getCoordinateSet().get(name)
        lo = float(coord.getRangeMin())
        hi = float(coord.getRangeMax())
    except Exception:
        return None
    if not np.isfinite(lo) or not np.isfinite(hi) or hi < lo:
        return None
    return lo, hi


def _sampled_pose(osim: Any, mapping: MappingConfig, rng: np.random.Generator, neutral_pose: dict[str, float]) -> dict[str, float]:
    cfg = _experiment(mapping)
    range_fraction = float(cfg.get("range_fraction", 0.35))
    min_width = float(cfg.get("min_width_rad", 1e-9))
    pose = dict(neutral_pose)
    for item in _independent_coordinate_items(osim, mapping):
        oname = mapping.side_name(item, "opensim")
        if not oname or oname.startswith("root_"):
            continue
        coord_range = _coordinate_range(osim, oname)
        if coord_range is None:
            continue
        lo, hi = coord_range
        if hi - lo <= min_width:
            continue
        center = float(neutral_pose.get(oname, osim.default_coordinates.get(oname, 0.0)))
        half_width = 0.5 * (hi - lo) * max(0.0, min(1.0, range_fraction))
        sample_lo = max(lo, center - half_width)
        sample_hi = min(hi, center + half_width)
        pose[oname] = float(rng.uniform(sample_lo, sample_hi))
    return pose


def _mujoco_pose_from_opensim_pose(opensim_pose: dict[str, float], mapping: MappingConfig) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in mapping.coordinates:
        oname = mapping.side_name(item, "opensim")
        mname = mapping.side_name(item, "mujoco")
        if not oname or not mname or mname == "root":
            continue
        if oname.startswith("root_"):
            continue
        if oname not in opensim_pose:
            continue
        sign = -1.0 if bool(item.get("sign_flip", False)) else 1.0
        values[mname] = sign * float(opensim_pose[oname])
    return values


def _combined_status(statuses: list[str]) -> str:
    if any(status == "failed" for status in statuses):
        return "failed"
    if any(status == "warning" for status in statuses):
        return "warning"
    if statuses and all(status == "passed" for status in statuses):
        return "passed"
    return "not evaluated"


def run(osim: Any, mjcf: Any, mapping: MappingConfig, out_dir: Path) -> dict[str, Any]:
    cfg = _experiment(mapping)
    sample_count = int(cfg.get("sample_count", 500))
    seed = int(cfg.get("seed", 20260508))
    rng = np.random.default_rng(seed)
    constraints = _coupler_constraints(osim)
    neutral_sample = next((s for s in mapping.pose_samples if str(s.get("name", "")) == "neutral"), mapping.pose_samples[0])
    neutral_pose = _with_coupled_coordinates(_pose_values(neutral_sample, mapping, "opensim"), constraints, mapping, "opensim")

    body_rows: list[dict[str, Any]] = []
    marker_rows: list[dict[str, Any]] = []
    orientation_rows: list[dict[str, Any]] = []
    for sample_index in range(sample_count):
        opensim_pose = _with_coupled_coordinates(_sampled_pose(osim, mapping, rng, neutral_pose), constraints, mapping, "opensim")
        mujoco_pose = _with_coupled_coordinates(_mujoco_pose_from_opensim_pose(opensim_pose, mapping), constraints, mapping, "mujoco")
        osim.set_pose(opensim_pose)
        mjcf.set_pose(mujoco_pose)
        for item in mapping.bodies:
            oname = mapping.side_name(item, "opensim")
            mname = mapping.side_name(item, "mujoco")
            if not oname or not mname:
                continue
            try:
                pos_err = norm_error(osim.body_position(oname), mjcf.body_position(mname))
                ori_err = _rotation_angle_error(osim.body_rotation_matrix(oname), mjcf.body_rotation_matrix(mname))
                status = "evaluated"
            except Exception as exc:
                pos_err = ori_err = np.nan
                status = f"skipped: {exc}"
            body_rows.append(
                {
                    "sample_index": sample_index,
                    "opensim_body": oname,
                    "mujoco_body": mname,
                    "position_error_m": pos_err,
                    "status": status,
                }
            )
            orientation_rows.append(
                {
                    "sample_index": sample_index,
                    "opensim_body": oname,
                    "mujoco_body": mname,
                    "orientation_error_rad": ori_err,
                    "orientation_error_deg": float(np.degrees(ori_err)) if np.isfinite(ori_err) else np.nan,
                    "status": status,
                }
            )
        for item in mapping.markers:
            oname = mapping.side_name(item, "opensim")
            mname = mapping.side_name(item, "mujoco")
            if not oname or not mname:
                continue
            try:
                err = norm_error(osim.marker_position(oname), mjcf.marker_position(mname))
                status = "evaluated"
            except Exception as exc:
                err = np.nan
                status = f"skipped: {exc}"
            marker_rows.append(
                {
                    "sample_index": sample_index,
                    "opensim_marker": oname,
                    "mujoco_site": mname,
                    "position_error_m": err,
                    "status": status,
                }
            )

    body_df = write_csv(out_dir / "random_fk_body_errors.csv", body_rows)
    marker_df = write_csv(out_dir / "random_fk_site_errors.csv", marker_rows)
    orientation_df = write_csv(out_dir / "random_fk_orientation_errors.csv", orientation_rows)
    files = ["random_fk_body_errors.csv", "random_fk_site_errors.csv", "random_fk_orientation_errors.csv", "random_fk_report.md"]
    for path, df, column in [
        ("random_fk_worst_body_error.csv", body_df, "position_error_m"),
        ("random_fk_worst_site_error.csv", marker_df, "position_error_m"),
        ("random_fk_worst_orientation_error.csv", orientation_df, "orientation_error_rad"),
    ]:
        worst = write_worst_csv(out_dir / "diagnostics" / path, df, column)
        if worst:
            files.append(f"diagnostics/{worst}")

    body_finite = [float(v) for v in body_df.get("position_error_m", []) if np.isfinite(v)]
    marker_finite = [float(v) for v in marker_df.get("position_error_m", []) if np.isfinite(v)]
    orientation_finite = [float(v) for v in orientation_df.get("orientation_error_rad", []) if np.isfinite(v)]
    pos_errors = body_finite + marker_finite
    pos_max = float(np.max(pos_errors)) if pos_errors else None
    ori_max = float(np.max(orientation_finite)) if orientation_finite else None
    pos_warn = float(mapping.thresholds.get("random_fk_position_warning_m", mapping.thresholds.get("position_warning_m", 0.01)))
    pos_fail = float(mapping.thresholds.get("random_fk_position_fail_m", mapping.thresholds.get("position_fail_m", 0.05)))
    ori_warn = float(mapping.thresholds.get("random_fk_orientation_warning_rad", mapping.thresholds.get("orientation_warning_rad", 1e-3)))
    ori_fail = float(mapping.thresholds.get("random_fk_orientation_fail_rad", mapping.thresholds.get("orientation_fail_rad", 1e-2)))
    pos_status = status_from_errors(pos_max, pos_warn, pos_fail)
    ori_status = status_from_errors(ori_max, ori_warn, ori_fail)
    status = _combined_status([pos_status, ori_status])

    write_markdown(
        out_dir / "random_fk_report.md",
        "\n".join(
            [
                "# Random FK Report",
                "",
                f"- sample_count: `{sample_count}`",
                f"- seed: `{seed}`",
                f"- coordinate_range_fraction: `{float(cfg.get('range_fraction', 0.35))}`",
                f"- max_position_error_m: `{pos_max}`",
                f"- max_orientation_error_rad: `{ori_max}`",
                f"- status: `{status}`",
                "",
                "This check samples independent OpenSim coordinates inside a centered fraction of their declared ranges, applies configured coordinate couplers, mirrors the pose through the mapping into MuJoCo, and compares body/site positions plus body orientations.",
            ]
        )
        + "\n",
    )
    return {
        "status": status,
        "sample_count": sample_count,
        "seed": seed,
        "evaluated_body_rows": len(body_finite),
        "evaluated_marker_rows": len(marker_finite),
        "evaluated_orientation_rows": len(orientation_finite),
        "max_position_error_m": pos_max,
        "rmse_position_error_m": rmse(pos_errors),
        "position_status": pos_status,
        "position_warning_threshold_m": pos_warn,
        "position_failure_threshold_m": pos_fail,
        "max_orientation_error_rad": ori_max,
        "max_orientation_error_deg": float(np.degrees(ori_max)) if ori_max is not None else None,
        "rmse_orientation_error_rad": rmse(orientation_finite),
        "orientation_status": ori_status,
        "orientation_warning_threshold_rad": ori_warn,
        "orientation_failure_threshold_rad": ori_fail,
        "note": "Random legal-pose FK check. It is stronger than neutral kinematics and joint sweep for hidden multi-joint frame/mapping errors.",
        "files": files,
    }
