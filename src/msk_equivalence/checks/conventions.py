from __future__ import annotations

from pathlib import Path
from typing import Any

from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import write_markdown


def run(osim: Any, mjcf: Any, mapping: MappingConfig, out_dir: Path) -> dict[str, Any]:
    conv = mapping.conventions
    rows = [
        "# Convention Report",
        "",
        "This check combines automatically readable metadata with required manual review items.",
        "",
        "## Automatically Readable",
        "",
        f"- OpenSim model: `{osim.path}`",
        f"- MuJoCo model: `{mjcf.path}`",
        f"- MuJoCo gravity: `{getattr(mjcf.model, 'opt', None).gravity.tolist() if hasattr(getattr(mjcf.model, 'opt', None), 'gravity') else 'unknown'}`",
        f"- MuJoCo compiler angle convention is resolved by MuJoCo at load time; qpos units are radians/meters.",
        "",
        "## Mapping-Specified Conventions",
        "",
    ]
    for key in ["length_unit", "mass_unit", "angle_unit", "forward_axis", "up_axis", "lateral_axis", "pelvis_root"]:
        rows.append(f"- {key}: `{conv.get(key, 'TODO')}`")
    rows.extend(
        [
            "",
            "## Manual Checks Required",
            "",
            "- Confirm OpenSim length and mass units from model provenance; OpenSim .osim files do not always encode this explicitly.",
            "- Confirm pelvis/root frame orientation and whether world axes match between engines.",
            "- For every coordinate, validate positive direction using the joint sweep report.",
            "- Mark `sign_flip: true` in `coordinates` mapping when q must be negated for equivalence.",
            "- Inertia comparison is risky if body frames or MuJoCo inertial frames are not aligned.",
        ]
    )
    write_markdown(out_dir / "convention_report.md", "\n".join(rows) + "\n")
    return {"status": "warning", "reason": "Several convention items require manual model-provenance validation.", "files": ["convention_report.md"]}
