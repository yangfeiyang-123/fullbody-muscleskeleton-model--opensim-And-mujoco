from __future__ import annotations

from pathlib import Path
from typing import Any

from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import write_csv, write_markdown


PARAMS = [
    "max_isometric_force",
    "optimal_fiber_length",
    "tendon_slack_length",
    "pennation_angle",
    "activation_time_constant",
    "deactivation_time_constant",
    "tendon_elasticity",
]


def run(osim: Any, mjcf: Any, mapping: MappingConfig, out_dir: Path) -> dict[str, Any]:
    rows = []
    for muscle in mapping.muscles:
        for param in PARAMS:
            rows.append(
                {
                    "opensim_muscle": mapping.side_name(muscle, "opensim"),
                    "mujoco_muscle_or_actuator": mapping.side_name(muscle, "mujoco"),
                    "parameter": param,
                    "opensim_value": "",
                    "mujoco_value": "",
                    "status": "not evaluated",
                    "note": "Backend-specific muscle parameter extraction requires model-class specific adapters.",
                }
            )
    write_csv(out_dir / "muscle_parameter_comparison.csv", rows)
    write_markdown(
        out_dir / "muscle_model_equivalence_notes.md",
        "# Muscle Model Equivalence Notes\n\n"
        "Parameter equivalence and function-output equivalence are separate checks.\n\n"
        "- Parameter equivalence compares declared constants such as max isometric force, optimal fiber length, tendon slack length and activation time constants.\n"
        "- Function-output equivalence compares force-length and force-velocity outputs over sampled states, which may match even when parameter names differ.\n"
        "- OpenSim and MuJoCo muscle models are often not identical; treat unmatched parameters as calibration targets rather than automatic failures.\n",
    )
    return {"status": "not evaluated", "reason": "Parameter extraction adapter is intentionally left explicit.", "files": ["muscle_parameter_comparison.csv", "muscle_model_equivalence_notes.md"]}
