from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from msk_equivalence.checks.kinematics import _pose_values
from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import correlation, ensure_dir, rel_error, status_from_errors, write_csv


def _plot(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plot_dir = ensure_dir(out_dir / "plots" / "muscle_length")
    muscles = sorted({r["opensim_muscle"] for r in rows})
    for muscle in muscles:
        subset = [r for r in rows if r["opensim_muscle"] == muscle and np.isfinite(r["opensim_length"]) and np.isfinite(r["mujoco_length"])]
        if not subset:
            continue
        x = list(range(len(subset)))
        plt.figure(figsize=(6, 4))
        plt.plot(x, [r["opensim_length"] for r in subset], label="OpenSim")
        plt.plot(x, [r["mujoco_length"] for r in subset], label="MuJoCo")
        plt.title(muscle)
        plt.xlabel("sample")
        plt.ylabel("muscle-tendon length")
        plt.legend()
        plt.tight_layout()
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in muscle)
        plt.savefig(plot_dir / f"{safe}.png", dpi=150)
        plt.close()


def run(osim: Any, mjcf: Any, mapping: MappingConfig, out_dir: Path) -> dict[str, Any]:
    rows = []
    for sample in mapping.pose_samples:
        sample_name = str(sample.get("name", "sample"))
        osim.set_pose(_pose_values(sample, mapping, "opensim"))
        mjcf.set_pose(_pose_values(sample, mapping, "mujoco"))
        for item in mapping.muscles:
            oname = mapping.side_name(item, "opensim")
            mname = mapping.side_name(item, "mujoco")
            try:
                ol = osim.muscle_length(oname)
                ml = mjcf.tendon_length(mname)
                abs_error = abs(ol - ml)
                status = "evaluated"
            except Exception as exc:
                ol = ml = abs_error = np.nan
                status = f"skipped: {exc}"
            rows.append(
                {
                    "sample": sample_name,
                    "opensim_muscle": oname,
                    "mujoco_tendon_or_actuator": mname,
                    "opensim_length": ol,
                    "mujoco_length": ml,
                    "absolute_error": abs_error,
                    "relative_error": rel_error(abs_error, ol),
                    "status": status,
                }
            )
    df = write_csv(out_dir / "muscle_length_error.csv", rows)
    if os.environ.get("MSK_EQUIVALENCE_SKIP_PLOTS") != "1":
        _plot(rows, out_dir)
    finite = df[df["absolute_error"].apply(np.isfinite)] if not df.empty else df
    max_error = float(finite["absolute_error"].max()) if not finite.empty else None
    warn = float(mapping.thresholds.get("muscle_length_warning_m", 0.005))
    fail = float(mapping.thresholds.get("muscle_length_fail_m", 0.05))
    return {
        "status": status_from_errors(max_error, warn, fail),
        "mean_abs_error": float(finite["absolute_error"].mean()) if not finite.empty else None,
        "max_error": max_error,
        "correlation": correlation(finite["opensim_length"], finite["mujoco_length"]) if not finite.empty else None,
        "warning_threshold_m": warn,
        "failure_threshold_m": fail,
        "files": ["muscle_length_error.csv", "plots/muscle_length/"],
    }
