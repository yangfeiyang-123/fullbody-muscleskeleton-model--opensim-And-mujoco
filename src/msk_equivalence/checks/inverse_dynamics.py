from __future__ import annotations

from pathlib import Path
from typing import Any

from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import write_csv


def _rows(mapping: MappingConfig) -> list[dict[str, Any]]:
    experiments = mapping.dynamics_experiments.get("inverse_dynamics", [])
    if not isinstance(experiments, list):
        return []
    rows = []
    for item in experiments:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "name": item.get("name", ""),
                "q_sample": item.get("q_sample", ""),
                "q_trajectory": item.get("q_trajectory", ""),
                "qdot": item.get("qdot", ""),
                "qddot": item.get("qddot", ""),
                "contacts": item.get("contacts", ""),
                "compare": item.get("compare", ""),
                "purpose": item.get("purpose", ""),
                "status": "planned",
            }
        )
    return rows


def run(osim: Any, mjcf: Any, mapping: MappingConfig, out_dir: Path) -> dict[str, Any]:
    motion = mapping.raw.get("motion_data", {})
    planned = _rows(mapping)
    if planned:
        write_csv(out_dir / "inverse_dynamics_plan.csv", planned)
    if not motion:
        reason = "No q(t), qdot(t), qddot(t) and external-force data configured in mapping.yaml."
    else:
        reason = "Motion data configured, but OpenSim/MuJoCo inverse dynamics adapters are not implemented in MVP."
    write_csv(out_dir / "inverse_dynamics_torque_error.csv", [{"status": "not evaluated", "reason": reason}])
    files = ["inverse_dynamics_torque_error.csv"]
    if planned:
        files.append("inverse_dynamics_plan.csv")
    return {"status": "not evaluated", "reason": reason, "planned_experiments": len(planned), "files": files}
