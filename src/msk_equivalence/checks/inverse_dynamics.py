from __future__ import annotations

from pathlib import Path
from typing import Any

from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import write_csv


def run(osim: Any, mjcf: Any, mapping: MappingConfig, out_dir: Path) -> dict[str, Any]:
    motion = mapping.raw.get("motion_data", {})
    if not motion:
        reason = "No q(t), qdot(t), qddot(t) and external-force data configured in mapping.yaml."
    else:
        reason = "Motion data configured, but OpenSim/MuJoCo inverse dynamics adapters are not implemented in MVP."
    write_csv(out_dir / "inverse_dynamics_torque_error.csv", [{"status": "not evaluated", "reason": reason}])
    return {"status": "not evaluated", "reason": reason, "files": ["inverse_dynamics_torque_error.csv"]}
