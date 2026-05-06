from __future__ import annotations

from pathlib import Path
from typing import Any

from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import write_csv


def run(osim: Any, mjcf: Any, mapping: MappingConfig, out_dir: Path) -> dict[str, Any]:
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
    return {"status": "not evaluated", "reason": "Generalized muscle torque adapters are not implemented in MVP.", "files": ["muscle_generated_torque_error.csv"]}
