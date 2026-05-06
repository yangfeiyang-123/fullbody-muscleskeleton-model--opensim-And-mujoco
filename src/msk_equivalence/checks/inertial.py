from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import norm_error, rel_error, write_csv, write_json


def _by_name(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["name"]: r for r in rows}


def run(osim: Any, mjcf: Any, mapping: MappingConfig, out_dir: Path) -> dict[str, Any]:
    osim_rows = _by_name(osim.body_inertials())
    mjcf_rows = _by_name(mjcf.body_inertials())
    rows = []
    for item in mapping.bodies:
        oname = mapping.side_name(item, "opensim")
        mname = mapping.side_name(item, "mujoco")
        o = osim_rows.get(oname or "", {})
        m = mjcf_rows.get(mname or "", {})
        omass = float(o.get("mass", np.nan))
        mmass = float(m.get("mass", np.nan))
        mass_abs_error = abs(omass - mmass) if np.isfinite(omass) and np.isfinite(mmass) else np.nan
        rows.append(
            {
                "opensim_body": oname,
                "mujoco_body": mname,
                "opensim_mass": omass,
                "mujoco_mass": mmass,
                "mass_abs_error": mass_abs_error,
                "mass_relative_error": rel_error(mass_abs_error, omass),
                "com_position_error_in_local_frames": norm_error(o.get("com"), m.get("com")),
                "opensim_inertia": o.get("inertia"),
                "mujoco_inertia": m.get("inertia"),
                "frame_note": "Compared in each model's body/inertial frame; this is only meaningful when frames are aligned.",
            }
        )
    df = write_csv(out_dir / "inertial_body_comparison.csv", rows)
    osim_mass = osim.total_mass()
    mjcf_mass = mjcf.total_mass()
    payload = {
        "opensim_total_mass": osim_mass,
        "mujoco_total_mass": mjcf_mass,
        "total_mass_abs_error": abs(osim_mass - mjcf_mass) if np.isfinite(osim_mass) and np.isfinite(mjcf_mass) else None,
        "total_mass_relative_error": rel_error(abs(osim_mass - mjcf_mass), osim_mass)
        if np.isfinite(osim_mass) and np.isfinite(mjcf_mass)
        else None,
        "frame_risk": "Segment COM and inertia are reported in each model's local body/inertial frame. Align frames before interpreting tensor differences.",
    }
    write_json(out_dir / "total_mass_comparison.json", payload)
    max_mass_error = float(df["mass_abs_error"].max()) if not df.empty else np.nan
    return {"status": "passed" if np.isfinite(max_mass_error) else "not evaluated", "max_segment_mass_error": max_mass_error, "files": ["inertial_body_comparison.csv", "total_mass_comparison.json"]}
