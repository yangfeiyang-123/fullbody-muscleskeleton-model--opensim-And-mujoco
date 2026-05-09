from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from msk_equivalence.adapters.runtime_dynamics import adapter_config
from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import write_csv, write_markdown


def _dataset_path(mapping: MappingConfig) -> Path | None:
    cfg = adapter_config(mapping)
    rl_cfg = cfg.get("rl_distribution", {}) if isinstance(cfg.get("rl_distribution", {}), dict) else {}
    value = rl_cfg.get("dataset") or rl_cfg.get("policy_rollout_dataset")
    return Path(value) if value else None


def _array_summary(path: Path) -> list[dict[str, Any]]:
    data = np.load(path, allow_pickle=False)
    rows: list[dict[str, Any]] = []
    for key in data.files:
        arr = data[key]
        rows.append(
            {
                "name": key,
                "shape": "x".join(str(v) for v in arr.shape),
                "dtype": str(arr.dtype),
                "finite_fraction": float(np.isfinite(arr).mean()) if np.issubdtype(arr.dtype, np.number) and arr.size else np.nan,
            }
        )
    return rows


def _has_required_arrays(rows: list[dict[str, Any]]) -> tuple[bool, str]:
    names = {str(row["name"]) for row in rows}
    action_names = {"actions", "action", "ctrl", "controls"}
    qpos_names = {"qpos", "positions", "state_qpos"}
    qvel_names = {"qvel", "velocities", "state_qvel"}
    missing = []
    if not names & action_names:
        missing.append("actions/ctrl")
    if not names & qpos_names:
        missing.append("qpos")
    if not names & qvel_names:
        missing.append("qvel")
    if missing:
        return False, "Dataset is missing required RL rollout arrays: " + ", ".join(missing)
    return True, ""


def run(osim: Any, mjcf: Any, mapping: MappingConfig, out_dir: Path) -> dict[str, Any]:
    path = _dataset_path(mapping)
    if path is None:
        write_markdown(
            out_dir / "rl_distribution_rollout_report.md",
            "# RL Distribution Rollout Gate\n\n"
            "status: `not evaluated`\n\n"
            "No MuJoCo policy rollout dataset was configured. This gate is required before claiming RL training-distribution equivalence.\n",
        )
        return {
            "status": "not evaluated",
            "reason": "No MuJoCo policy rollout dataset configured in adapter config.",
            "note": "Provide a policy rollout dataset before claiming RL distribution equivalence.",
            "files": ["rl_distribution_rollout_report.md"],
        }
    if not path.exists():
        write_markdown(
            out_dir / "rl_distribution_rollout_report.md",
            f"# RL Distribution Rollout Gate\n\nstatus: `not evaluated`\n\nConfigured dataset does not exist: `{path}`.\n",
        )
        return {
            "status": "not evaluated",
            "reason": f"Configured MuJoCo policy rollout dataset does not exist: {path}",
            "files": ["rl_distribution_rollout_report.md"],
        }
    try:
        rows = _array_summary(path)
    except Exception as exc:
        return {"status": "failed", "reason": f"Could not inspect RL rollout dataset: {exc}", "files": []}
    write_csv(out_dir / "diagnostics" / "rl_distribution_dataset_summary.csv", rows)
    has_required, reason = _has_required_arrays(rows)
    status = "not evaluated"
    if has_required:
        reason = (
            "RL rollout dataset schema is present, but runtime OpenSim+adapter replay is not implemented for this dataset schema yet. "
            "This gate must be completed before claiming RL distribution equivalence."
        )
    write_markdown(
        out_dir / "rl_distribution_rollout_report.md",
        "\n".join(
            [
                "# RL Distribution Rollout Gate",
                "",
                f"- status: `{status}`",
                f"- dataset: `{path}`",
                f"- reason: {reason}",
                "",
                "This gate is intentionally conservative: dataset presence alone is not accepted as RL equivalence evidence.",
            ]
        )
        + "\n",
    )
    return {
        "status": status,
        "reason": reason,
        "dataset": str(path),
        "array_count": len(rows),
        "note": "Dataset inspection only. Actual MuJoCo-policy OpenSim+adapter replay remains required.",
        "files": ["rl_distribution_rollout_report.md", "diagnostics/rl_distribution_dataset_summary.csv"],
    }
