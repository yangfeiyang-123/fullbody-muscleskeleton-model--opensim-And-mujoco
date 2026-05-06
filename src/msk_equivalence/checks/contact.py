from __future__ import annotations

from pathlib import Path
from typing import Any

from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import write_csv, write_markdown


def run(osim: Any, mjcf: Any, mapping: MappingConfig, out_dir: Path) -> dict[str, Any]:
    rows = []
    for item in mapping.contacts:
        rows.append(
            {
                "opensim_contact": mapping.side_name(item, "opensim"),
                "mujoco_contact": mapping.side_name(item, "mujoco"),
                "status": "manual review",
                "note": "Compare geometry, stiffness, damping and friction directly only when contact formulations are compatible.",
            }
        )
    if not rows:
        rows = [{"status": "not evaluated", "reason": "No contacts mapped in mapping.yaml."}]
    write_csv(out_dir / "contact_model_comparison.csv", rows)
    write_markdown(
        out_dir / "contact_notes.md",
        "# Contact Notes\n\n"
        "OpenSim and MuJoCo contact models are usually not strictly parameter-equivalent.\n\n"
        "- Direct comparison: geometry names, ground plane, nominal friction and configured stiffness/damping when present.\n"
        "- Behavior comparison: contact point, GRF, CoP, penetration depth and sliding velocity under matched motions.\n"
        "- Manual tuning: stiffness/damping/friction often need task-level calibration for standing, landing and running.\n",
    )
    return {"status": "warning", "reason": "Contact equivalence normally requires behavior-level validation.", "files": ["contact_model_comparison.csv", "contact_notes.md"]}
