from __future__ import annotations

from pathlib import Path
from typing import Any

from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import write_csv


def run(osim: Any, mjcf: Any, mapping: MappingConfig, out_dir: Path) -> dict[str, Any]:
    reason = "Requires explicit adapters for passive muscle, ligament, limit, damping and stiffness force terms."
    write_csv(out_dir / "passive_force_comparison.csv", [{"status": "not evaluated", "reason": reason}])
    write_csv(out_dir / "ligament_comparison.csv", [{"status": "not evaluated", "reason": reason}])
    return {"status": "not evaluated", "reason": reason, "files": ["passive_force_comparison.csv", "ligament_comparison.csv"]}
