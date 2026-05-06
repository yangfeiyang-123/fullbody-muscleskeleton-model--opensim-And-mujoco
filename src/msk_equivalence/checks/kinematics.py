from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import norm_error, rmse, write_csv


def _pose_values(sample: dict[str, Any], mapping: MappingConfig, side: str) -> dict[str, float]:
    raw_q = sample.get("q", {})
    if not isinstance(raw_q, dict):
        return {}
    values: dict[str, float] = {}
    for coord in mapping.coordinates:
        oname = mapping.side_name(coord, "opensim")
        mname = mapping.side_name(coord, "mujoco")
        target = oname if side == "opensim" else mname
        source = oname or mname
        if not target or not source:
            continue
        if source in raw_q:
            val = float(raw_q[source])
        elif target in raw_q:
            val = float(raw_q[target])
        else:
            continue
        sign = -1.0 if bool(coord.get("sign_flip", False)) and side == "mujoco" else 1.0
        values[target] = sign * val
    return values


def run(osim: Any, mjcf: Any, mapping: MappingConfig, out_dir: Path) -> dict[str, Any]:
    body_rows = []
    marker_rows = []
    com_rows = []
    for sample in mapping.pose_samples:
        name = str(sample.get("name", "sample"))
        osim.set_pose(_pose_values(sample, mapping, "opensim"))
        mjcf.set_pose(_pose_values(sample, mapping, "mujoco"))
        for item in mapping.bodies:
            oname = mapping.side_name(item, "opensim")
            mname = mapping.side_name(item, "mujoco")
            try:
                err = norm_error(osim.body_position(oname), mjcf.body_position(mname))
                status = "evaluated"
            except Exception as exc:
                err = np.nan
                status = f"skipped: {exc}"
            body_rows.append({"sample": name, "opensim_body": oname, "mujoco_body": mname, "position_error": err, "status": status})
        for item in mapping.markers:
            oname = mapping.side_name(item, "opensim")
            mname = mapping.side_name(item, "mujoco")
            try:
                err = norm_error(osim.marker_position(oname), mjcf.marker_position(mname))
                status = "evaluated"
            except Exception as exc:
                err = np.nan
                status = f"skipped: {exc}"
            marker_rows.append({"sample": name, "opensim_marker": oname, "mujoco_site": mname, "position_error": err, "status": status})
        try:
            com_err = norm_error(osim.whole_body_com(), mjcf.whole_body_com())
            status = "evaluated"
        except Exception as exc:
            com_err = np.nan
            status = f"skipped: {exc}"
        com_rows.append({"sample": name, "whole_body_com_error": com_err, "status": status})
    body_df = write_csv(out_dir / "kinematics_body_pose_error.csv", body_rows)
    marker_df = write_csv(out_dir / "kinematics_marker_site_error.csv", marker_rows)
    com_df = write_csv(out_dir / "whole_body_com_error.csv", com_rows)
    errors = list(body_df.get("position_error", [])) + list(marker_df.get("position_error", [])) + list(com_df.get("whole_body_com_error", []))
    finite = [float(v) for v in errors if np.isfinite(v)]
    return {
        "status": "passed" if finite else "not evaluated",
        "mean_error": float(np.mean(finite)) if finite else None,
        "max_error": float(np.max(finite)) if finite else None,
        "rmse": rmse(finite),
        "files": ["kinematics_body_pose_error.csv", "kinematics_marker_site_error.csv", "whole_body_com_error.csv"],
    }
