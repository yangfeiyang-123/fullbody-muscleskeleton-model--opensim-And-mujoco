from __future__ import annotations

from pathlib import Path
from typing import Any

from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import write_markdown


def run(osim: Any, mjcf: Any, mapping: MappingConfig, out_dir: Path) -> dict[str, Any]:
    conv = mapping.conventions
    alignment = mapping.raw.get("frame_alignment", {})
    required = ["length_unit", "mass_unit", "angle_unit", "forward_axis", "up_axis", "lateral_axis", "pelvis_root"]
    missing = [key for key in required if str(conv.get(key, "TODO")).startswith("TODO")]
    has_alignment = isinstance(alignment, dict) and bool(alignment.get("opensim_to_mujoco_rotation"))
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
            "## Frame Alignment",
            "",
            f"- OpenSim to MuJoCo rotation: `{alignment.get('opensim_to_mujoco_rotation', 'TODO') if isinstance(alignment, dict) else 'TODO'}`",
            f"- note: `{alignment.get('note', '') if isinstance(alignment, dict) else ''}`",
        ]
    )
    rows.extend(
        [
            "",
            "## Residual Checks",
            "",
            "- Coordinate positive directions are validated by the joint sweep report.",
            "- Inertia comparison uses the explicit frame-alignment metadata and the inertial gate.",
        ]
    )
    if missing:
        rows.extend(["", "## Missing Convention Metadata", ""])
        rows.extend(f"- {key}" for key in missing)
    write_markdown(out_dir / "convention_report.md", "\n".join(rows) + "\n")
    if missing or not has_alignment:
        return {
            "status": "warning",
            "reason": "Convention metadata is incomplete or frame alignment is missing.",
            "files": ["convention_report.md"],
        }
    return {"status": "passed", "files": ["convention_report.md"]}
