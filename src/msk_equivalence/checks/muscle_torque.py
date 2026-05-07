from __future__ import annotations

from pathlib import Path
from typing import Any

from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import write_csv


def _plan_rows(mapping: MappingConfig) -> list[dict[str, Any]]:
    experiments = mapping.dynamics_experiments.get("muscle_torque", [])
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
                "activation_values": item.get("activation_values", ""),
                "groups": item.get("groups", ""),
                "compare": item.get("compare", ""),
                "purpose": item.get("purpose", ""),
                "status": "planned",
            }
        )
    return rows


def run(osim: Any, mjcf: Any, mapping: MappingConfig, out_dir: Path) -> dict[str, Any]:
    planned = _plan_rows(mapping)
    if planned:
        write_csv(out_dir / "muscle_torque_plan.csv", planned)
    write_csv(
        out_dir / "muscle_generated_torque_error.csv",
        [
            {
                "test": "single_muscle_activation",
                "status": "not evaluated",
                "reason": "Requires backend-specific generalized-force adapters for OpenSim and MuJoCo controls.",
            },
            {
                "test": "muscle_group_activation",
                "status": "not evaluated",
                "reason": "Requires a muscle group definition in mapping.yaml.",
            },
            {
                "test": "random_activation_vector",
                "status": "not evaluated",
                "reason": "Requires consistent activation/control ordering.",
            },
        ],
    )
    files = ["muscle_generated_torque_error.csv"]
    if planned:
        files.append("muscle_torque_plan.csv")
    return {
        "status": "not evaluated",
        "reason": "Generalized muscle torque adapters are not implemented in MVP.",
        "planned_experiments": len(planned),
        "files": files,
    }
