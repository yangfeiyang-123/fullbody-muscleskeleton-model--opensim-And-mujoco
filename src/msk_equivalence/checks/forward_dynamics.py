from __future__ import annotations

from pathlib import Path
from typing import Any

from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import write_csv, write_markdown


def run(osim: Any, mjcf: Any, mapping: MappingConfig, out_dir: Path) -> dict[str, Any]:
    reason = "Short forward dynamics smoke tests require synchronized state/control adapters and are not implemented in MVP."
    write_csv(out_dir / "forward_dynamics_smoke_test.csv", [{"test": "passive_drop_0.1s", "status": "not evaluated", "reason": reason}])
    write_markdown(
        out_dir / "forward_dynamics_notes.md",
        "# Forward Dynamics Notes\n\n"
        "Do not use long simulations as an equivalence test by default; small numerical differences can diverge quickly.\n"
        "Recommended smoke tests are passive drop for 0.1 s, a single joint torque pulse, and a 0.2 s matched-control rollout.\n",
    )
    return {"status": "not evaluated", "reason": reason, "files": ["forward_dynamics_smoke_test.csv", "forward_dynamics_notes.md"]}
