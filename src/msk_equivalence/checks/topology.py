from __future__ import annotations

from pathlib import Path
from typing import Any

from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import write_csv, write_json


def _mapped_missing(mapped: list[str], available: list[str]) -> list[str]:
    available_set = set(available)
    return sorted([name for name in mapped if name not in available_set])


def run(osim: Any, mjcf: Any, mapping: MappingConfig, out_dir: Path) -> dict[str, Any]:
    osim_counts = {
        "bodies": len(osim.bodies),
        "coordinates": len(osim.coordinates),
        "muscles": len(osim.muscles),
        "actuators": len(osim.actuators),
        "markers_sites": len(osim.markers),
    }
    mjcf_counts = {
        "bodies": len(mjcf.bodies),
        "coordinates": len(mjcf.coordinates),
        "muscles": len(mjcf.muscles),
        "actuators": len(mjcf.actuators),
        "markers_sites": len(mjcf.markers),
    }
    rows = []
    for key in sorted(osim_counts):
        rows.append(
            {
                "entity": key,
                "opensim_count": osim_counts[key],
                "mujoco_count": mjcf_counts[key],
                "difference": osim_counts[key] - mjcf_counts[key],
            }
        )
    write_csv(out_dir / "topology_summary.csv", rows)

    mismatch = {
        "mapped_opensim_missing": {
            "bodies": _mapped_missing(mapping.opensim_names("bodies"), osim.bodies),
            "coordinates": _mapped_missing(mapping.opensim_names("coordinates"), osim.coordinates),
            "muscles": _mapped_missing(mapping.opensim_names("muscles"), osim.muscles),
            "markers": _mapped_missing(mapping.opensim_names("markers"), osim.markers),
        },
        "mapped_mujoco_missing": {
            "bodies": _mapped_missing(mapping.mujoco_names("bodies"), mjcf.bodies),
            "coordinates": _mapped_missing(mapping.mujoco_names("coordinates"), mjcf.coordinates),
            "muscles": _mapped_missing(mapping.mujoco_names("muscles"), mjcf.muscles),
            "sites": _mapped_missing(mapping.mujoco_names("markers"), mjcf.markers),
        },
        "mapped_only": {
            "opensim_bodies_without_mujoco_mapping": sorted(set(osim.bodies) - set(mapping.opensim_names("bodies"))),
            "mujoco_bodies_without_opensim_mapping": sorted(set(mjcf.bodies) - set(mapping.mujoco_names("bodies"))),
            "opensim_coordinates_without_mujoco_mapping": sorted(set(osim.coordinates) - set(mapping.opensim_names("coordinates"))),
            "mujoco_coordinates_without_opensim_mapping": sorted(set(mjcf.coordinates) - set(mapping.mujoco_names("coordinates"))),
            "opensim_muscles_without_mujoco_mapping": sorted(set(osim.muscles) - set(mapping.opensim_names("muscles"))),
            "mujoco_muscles_without_opensim_mapping": sorted(set(mjcf.muscles) - set(mapping.mujoco_names("muscles"))),
        },
        "hierarchy": {
            "opensim": osim.hierarchy(),
            "mujoco": mjcf.hierarchy(),
            "note": "Hierarchy equality is checked by mapped names only; inspect parent/child rows for frame-level differences.",
        },
        "root_free_node": {
            "opensim_has_root_translation_rotation_coordinates": all(
                name in osim.coordinates for name in ["root_tx", "root_ty", "root_tz", "root_rx", "root_ry", "root_rz"]
            ),
            "mujoco_has_free_root": "root" in mjcf.joints,
        },
    }
    write_json(out_dir / "topology_mismatch.json", mismatch)
    missing_count = sum(len(v) for side in ["mapped_opensim_missing", "mapped_mujoco_missing"] for v in mismatch[side].values())
    status = "passed" if missing_count == 0 else "warning"
    return {"status": status, "missing_mapped_entities": missing_count, "files": ["topology_summary.csv", "topology_mismatch.json"]}
