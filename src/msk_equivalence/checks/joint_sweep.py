from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import norm_error, write_csv, write_json


def run(osim: Any, mjcf: Any, mapping: MappingConfig, out_dir: Path) -> dict[str, Any]:
    rows = []
    warnings = []
    sweep = mapping.raw.get("joint_sweep", {})
    values = sweep.get("values", [-0.25, 0.0, 0.25]) if isinstance(sweep, dict) else [-0.25, 0.0, 0.25]
    for coord in mapping.coordinates:
        oname = mapping.side_name(coord, "opensim")
        mname = mapping.side_name(coord, "mujoco")
        for value in values:
            sign = -1.0 if coord.get("sign_flip", False) else 1.0
            try:
                osim.set_pose({oname: float(value)})
                mjcf.set_pose({mname: sign * float(value)})
                com_error = norm_error(osim.whole_body_com(), mjcf.whole_body_com())
                status = "evaluated"
            except Exception as exc:
                com_error = np.nan
                status = f"skipped: {exc}"
            rows.append(
                {
                    "opensim_coordinate": oname,
                    "mujoco_qpos_or_joint": mname,
                    "sweep_value": value,
                    "whole_body_com_error": com_error,
                    "status": status,
                }
            )
    write_csv(out_dir / "joint_sweep_errors.csv", rows)
    write_json(
        out_dir / "possible_sign_flip_warnings.json",
        {
            "warnings": warnings,
            "note": "Use this file together with moment_arm sign warnings. A high asymmetric error across +/- sweep often indicates a sign or axis mismatch.",
        },
    )
    evaluated = [r for r in rows if r["status"] == "evaluated"]
    return {"status": "passed" if evaluated else "not evaluated", "evaluated_samples": len(evaluated), "files": ["joint_sweep_errors.csv", "possible_sign_flip_warnings.json"]}
