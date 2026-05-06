from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from msk_equivalence.checks.kinematics import _pose_values
from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import correlation, ensure_dir, rmse, write_csv, write_json


def _plot(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plot_dir = ensure_dir(out_dir / "plots" / "moment_arm")
    pairs = sorted({(r["opensim_muscle"], r["opensim_coordinate"]) for r in rows})
    for muscle, coord in pairs:
        subset = [r for r in rows if r["opensim_muscle"] == muscle and r["opensim_coordinate"] == coord and np.isfinite(r["opensim_moment_arm"]) and np.isfinite(r["mujoco_moment_arm"])]
        if not subset:
            continue
        x = list(range(len(subset)))
        plt.figure(figsize=(6, 4))
        plt.plot(x, [r["opensim_moment_arm"] for r in subset], label="OpenSim")
        plt.plot(x, [r["mujoco_moment_arm"] for r in subset], label="MuJoCo numeric")
        plt.title(f"{muscle} / {coord}")
        plt.xlabel("sample")
        plt.ylabel("moment arm")
        plt.legend()
        plt.tight_layout()
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in f"{muscle}_{coord}")
        plt.savefig(plot_dir / f"{safe}.png", dpi=150)
        plt.close()


def _pairs(mapping: MappingConfig) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    explicit = mapping.raw.get("moment_arm_pairs")
    if isinstance(explicit, list) and explicit:
        pairs = []
        muscles_by_o = {mapping.side_name(m, "opensim"): m for m in mapping.muscles}
        coords_by_o = {mapping.side_name(c, "opensim"): c for c in mapping.coordinates}
        for item in explicit:
            if not isinstance(item, dict):
                continue
            muscle = muscles_by_o.get(item.get("muscle"))
            coord = coords_by_o.get(item.get("coordinate"))
            if muscle and coord:
                pairs.append((muscle, coord))
        return pairs
    return [(m, c) for m in mapping.muscles for c in mapping.coordinates]


def run(osim: Any, mjcf: Any, mapping: MappingConfig, out_dir: Path) -> dict[str, Any]:
    eps = float(mapping.thresholds.get("moment_arm_fd_epsilon", 1e-6))
    rows = []
    warnings = []
    for sample in mapping.pose_samples:
        sample_name = str(sample.get("name", "sample"))
        osim.set_pose(_pose_values(sample, mapping, "opensim"))
        mjcf.set_pose(_pose_values(sample, mapping, "mujoco"))
        for muscle, coord in _pairs(mapping):
            om = mapping.side_name(muscle, "opensim")
            mt = mapping.side_name(muscle, "mujoco")
            oc = mapping.side_name(coord, "opensim")
            mq = mapping.side_name(coord, "mujoco")
            try:
                ora = osim.moment_arm(om, oc)
                mra = mjcf.moment_arm_numeric(mt, mq, eps=eps)
                err = abs(ora - mra)
                sign_consistent = np.sign(ora) == np.sign(mra) or abs(ora) < 1e-9 or abs(mra) < 1e-9
                status = "evaluated"
                if not sign_consistent:
                    warnings.append({"sample": sample_name, "muscle": om, "coordinate": oc, "opensim": ora, "mujoco": mra})
            except Exception as exc:
                ora = mra = err = np.nan
                sign_consistent = None
                status = f"skipped: {exc}"
            rows.append(
                {
                    "sample": sample_name,
                    "opensim_muscle": om,
                    "mujoco_tendon_or_actuator": mt,
                    "opensim_coordinate": oc,
                    "mujoco_qpos_or_joint": mq,
                    "opensim_moment_arm": ora,
                    "mujoco_moment_arm": mra,
                    "absolute_error": err,
                    "sign_consistent": sign_consistent,
                    "status": status,
                }
            )
    df = write_csv(out_dir / "moment_arm_error.csv", rows)
    write_json(out_dir / "moment_arm_sign_warnings.json", {"warnings": warnings})
    _plot(rows, out_dir)
    finite = df[df["absolute_error"].apply(np.isfinite)] if not df.empty else df
    return {
        "status": "warning" if warnings else ("passed" if not finite.empty else "not evaluated"),
        "max_error": float(finite["absolute_error"].max()) if not finite.empty else None,
        "rmse": rmse(finite["absolute_error"]) if not finite.empty else None,
        "correlation": correlation(finite["opensim_moment_arm"], finite["mujoco_moment_arm"]) if not finite.empty else None,
        "sign_warning_count": len(warnings),
        "files": ["moment_arm_error.csv", "moment_arm_sign_warnings.json", "plots/moment_arm/"],
    }
