from __future__ import annotations

from pathlib import Path
from typing import Any

from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import write_csv


def _plan_rows(mapping: MappingConfig) -> list[dict[str, Any]]:
    experiments = mapping.dynamics_experiments.get("passive_forces", [])
    if not isinstance(experiments, list):
        return []
    rows = []
    for item in experiments:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "name": item.get("name", ""),
                "q_samples": item.get("q_samples", ""),
                "qdot": item.get("qdot", ""),
                "activation": item.get("activation", ""),
                "compare": item.get("compare", ""),
                "purpose": item.get("purpose", ""),
                "status": "planned",
            }
        )
    return rows


def run(osim: Any, mjcf: Any, mapping: MappingConfig, out_dir: Path) -> dict[str, Any]:
    planned = _plan_rows(mapping)
    if planned:
        write_csv(out_dir / "passive_force_plan.csv", planned)
    reason = "Requires explicit adapters for passive muscle, ligament, limit, damping and stiffness force terms."
    write_csv(out_dir / "passive_force_comparison.csv", [{"status": "not evaluated", "reason": reason}])
    write_csv(out_dir / "ligament_comparison.csv", [{"status": "not evaluated", "reason": reason}])
    files = ["passive_force_comparison.csv", "ligament_comparison.csv"]
    if planned:
        files.append("passive_force_plan.csv")
    return {"status": "not evaluated", "reason": reason, "planned_experiments": len(planned), "files": files}
