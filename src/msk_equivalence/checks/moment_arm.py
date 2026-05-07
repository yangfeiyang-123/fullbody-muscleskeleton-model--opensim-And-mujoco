from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from msk_equivalence.checks.kinematics import _pose_values
from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import correlation, ensure_dir, rmse, write_csv, write_json, write_worst_csv


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


def _dependent_coordinates(osim: Any) -> set[str]:
    path = getattr(osim, "path", None)
    if path is None:
        return set()
    try:
        text = Path(path).read_text(encoding="utf-8")
    except Exception:
        return set()
    return {item.strip() for item in re.findall(r"<dependent_coordinate_name>(.*?)</dependent_coordinate_name>", text)}


def _status(max_error: float | None, corr: float, sign_warnings: int, mapping: MappingConfig) -> str:
    if max_error is None or not np.isfinite(max_error):
        return "not evaluated"
    fail = float(mapping.thresholds.get("moment_arm_independent_fail_m", 0.1))
    warn = float(mapping.thresholds.get("moment_arm_independent_warning_m", mapping.thresholds.get("moment_arm_warning_m", 0.005)))
    corr_fail = float(mapping.thresholds.get("moment_arm_correlation_fail", 0.9))
    corr_warn = float(mapping.thresholds.get("moment_arm_correlation_warning", 0.98))
    if max_error > fail or (np.isfinite(corr) and corr < corr_fail):
        return "failed"
    if max_error > warn or sign_warnings > 0 or (np.isfinite(corr) and corr < corr_warn):
        return "warning"
    return "passed"


def run(osim: Any, mjcf: Any, mapping: MappingConfig, out_dir: Path) -> dict[str, Any]:
    eps = float(mapping.thresholds.get("moment_arm_fd_epsilon", 1e-6))
    dependent = _dependent_coordinates(osim)
    rows = []
    warnings = []
    direct_warnings = []
    for sample in mapping.pose_samples:
        sample_name = str(sample.get("name", "sample"))
        osim.set_pose(_pose_values(sample, mapping, "opensim"))
        mjcf.set_pose(_pose_values(sample, mapping, "mujoco"))
        for muscle, coord in _pairs(mapping):
            om = mapping.side_name(muscle, "opensim")
            mt = mapping.side_name(muscle, "mujoco")
            oc = mapping.side_name(coord, "opensim")
            mq = mapping.side_name(coord, "mujoco")
            role = "dependent" if oc in dependent else "independent"
            try:
                ora = osim.moment_arm(om, oc)
                mra = mjcf.moment_arm_numeric(mt, mq, eps=eps)
                err = abs(ora - mra)
                sign_consistent = np.sign(ora) == np.sign(mra) or abs(ora) < 1e-9 or abs(mra) < 1e-9
                status = "evaluated"
                if not sign_consistent:
                    item = {"sample": sample_name, "muscle": om, "coordinate": oc, "role": role, "opensim": ora, "mujoco": mra}
                    warnings.append(item)
                    if role == "independent":
                        direct_warnings.append(item)
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
                    "coordinate_role": role,
                    "status": status,
                }
            )
    df = write_csv(out_dir / "moment_arm_error.csv", rows)
    write_json(out_dir / "moment_arm_sign_warnings.json", {"warnings": warnings})
    files = ["moment_arm_error.csv", "moment_arm_sign_warnings.json", "plots/moment_arm/"]
    worst = write_worst_csv(out_dir / "diagnostics" / "moment_arm_worst_error.csv", df, "absolute_error")
    if worst:
        files.append(f"diagnostics/{worst}")
    if os.environ.get("MSK_EQUIVALENCE_SKIP_PLOTS") != "1":
        _plot(rows, out_dir)
    finite = df[df["absolute_error"].apply(np.isfinite)] if not df.empty else df
    direct = finite[finite["coordinate_role"] == "independent"] if not finite.empty else finite
    dependent_rows = finite[finite["coordinate_role"] == "dependent"] if not finite.empty else finite
    direct_worst = write_worst_csv(out_dir / "diagnostics" / "moment_arm_worst_independent_error.csv", direct, "absolute_error")
    if direct_worst:
        files.append(f"diagnostics/{direct_worst}")
    direct_corr = correlation(direct["opensim_moment_arm"], direct["mujoco_moment_arm"]) if not direct.empty else np.nan
    direct_max = float(direct["absolute_error"].max()) if not direct.empty else None
    direct_sign_count = len(direct_warnings)
    return {
        "status": _status(direct_max, direct_corr, direct_sign_count, mapping),
        "max_error": float(finite["absolute_error"].max()) if not finite.empty else None,
        "rmse": rmse(finite["absolute_error"]) if not finite.empty else None,
        "correlation": correlation(finite["opensim_moment_arm"], finite["mujoco_moment_arm"]) if not finite.empty else None,
        "sign_warning_count": len(warnings),
        "direct_independent_max_error": direct_max,
        "direct_independent_rmse": rmse(direct["absolute_error"]) if not direct.empty else None,
        "direct_independent_correlation": direct_corr,
        "direct_independent_sign_warning_count": direct_sign_count,
        "dependent_coordinate_pair_count": int(len(dependent_rows)),
        "note": "Status gates direct independent-coordinate moment arms only. Dependent coordinates are recorded but require constraint-chain-aware validation.",
        "files": files,
    }
