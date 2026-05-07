from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import write_csv, write_markdown


def _opensim_contact_names(osim: Any) -> set[str]:
    path = getattr(osim, "path", None)
    if path is None:
        return set()
    try:
        text = Path(path).read_text(encoding="utf-8")
    except Exception:
        return set()
    match = re.search(r"<ContactGeometrySet\b.*?</ContactGeometrySet>", text, re.DOTALL)
    if not match:
        return set()
    return set(
        re.findall(
            r'<(?:ContactSphere|ContactHalfSpace|ContactMesh|ContactCylinder|ContactEllipsoid|ContactTorus|ContactGeometry)\s+name="([^"]+)"',
            match.group(0),
        )
    )


def _pair_names(item: dict[str, Any], side: str) -> tuple[str | None, str | None]:
    value = item.get(side)
    if isinstance(value, dict):
        return value.get("geom1") or value.get("name") or value.get("geom"), value.get("geom2")
    if value is None:
        return None, None
    text = str(value)
    if "," in text:
        left, right = text.split(",", 1)
        return left.strip(), right.strip()
    return text, None


def _missing_contact(name: str | None, opensim_contacts: set[str]) -> bool:
    return not name or name.startswith("TODO_") or name not in opensim_contacts


def run(osim: Any, mjcf: Any, mapping: MappingConfig, out_dir: Path) -> dict[str, Any]:
    opensim_contacts = _opensim_contact_names(osim)
    rows = []
    failed = 0
    for item in mapping.contacts:
        ogeom1, ogeom2 = _pair_names(item, "opensim")
        mgeom1, mgeom2 = _pair_names(item, "mujoco")
        missing_or_todo = _missing_contact(ogeom1, opensim_contacts) or _missing_contact(ogeom2, opensim_contacts)
        status = "failed" if missing_or_todo else "mapped"
        if status == "failed":
            failed += 1
        rows.append(
            {
                "opensim_geom1": ogeom1,
                "opensim_geom2": ogeom2,
                "mujoco_geom1": mgeom1,
                "mujoco_geom2": mgeom2,
                "status": status,
                "note": "One or both OpenSim contact geometries are missing or still TODO."
                if missing_or_todo
                else "Mapped contact pair inventory entry.",
            }
        )
    if not rows:
        rows = [{"status": "not evaluated", "reason": "No contacts mapped in mapping.yaml."}]
    write_csv(out_dir / "contact_model_comparison.csv", rows)
    write_markdown(
        out_dir / "contact_notes.md",
        "# Contact Notes\n\n"
        "OpenSim and MuJoCo contact models are usually not strictly parameter-equivalent.\n\n"
        f"- OpenSim contact geometry count: `{len(opensim_contacts)}`.\n"
        f"- Failed mapped/TODO contact entries: `{failed}`.\n"
        "- Direct comparison: geometry names, ground plane, nominal friction and configured stiffness/damping when present.\n"
        "- Behavior comparison: contact point, GRF, CoP, penetration depth and sliding velocity under matched motions.\n"
        "- Manual tuning: stiffness/damping/friction often need task-level calibration for standing, landing and running.\n",
    )
    if failed:
        return {
            "status": "failed",
            "reason": "OpenSim contact geometry is missing or TODO for mapped MuJoCo contact pairs.",
            "failed_mapped_contacts": failed,
            "opensim_contact_geometry_count": len(opensim_contacts),
            "files": ["contact_model_comparison.csv", "contact_notes.md"],
        }
    return {
        "status": "passed" if rows and rows[0].get("status") != "not evaluated" else "not evaluated",
        "opensim_contact_geometry_count": len(opensim_contacts),
        "files": ["contact_model_comparison.csv", "contact_notes.md"],
    }
