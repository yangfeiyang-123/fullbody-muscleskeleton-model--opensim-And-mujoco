from __future__ import annotations

from pathlib import Path
from typing import Any

from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import write_csv, write_markdown


def _plan_rows(mapping: MappingConfig) -> list[dict[str, Any]]:
    experiments = mapping.dynamics_experiments.get("forward_dynamics", [])
    if not isinstance(experiments, list):
        return []
    rows = []
    for item in experiments:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "name": item.get("name", ""),
                "horizon_s": item.get("horizon_s", ""),
                "dt_s": item.get("dt_s", ""),
                "q_sample": item.get("q_sample", ""),
                "qdot": item.get("qdot", ""),
                "activation": item.get("activation", ""),
                "contacts": item.get("contacts", ""),
                "compare": item.get("compare", ""),
                "purpose": item.get("purpose", ""),
                "status": "planned",
            }
        )
    return rows


def run(osim: Any, mjcf: Any, mapping: MappingConfig, out_dir: Path) -> dict[str, Any]:
    planned = _plan_rows(mapping)
    if planned:
        write_csv(out_dir / "forward_dynamics_plan.csv", planned)
    reason = "Short forward dynamics smoke tests require synchronized state/control adapters and are not implemented in MVP."
    write_csv(out_dir / "forward_dynamics_smoke_test.csv", [{"test": "passive_drop_0.1s", "status": "not evaluated", "reason": reason}])
    write_markdown(
        out_dir / "forward_dynamics_notes.md",
        "# Forward Dynamics Notes\n\n"
        "Do not use long simulations as an equivalence test by default; small numerical differences can diverge quickly.\n"
        "Recommended smoke tests are passive drop for 0.1 s, a single joint torque pulse, and a 0.2 s matched-control rollout.\n",
    )
    files = ["forward_dynamics_smoke_test.csv", "forward_dynamics_notes.md"]
    if planned:
        files.append("forward_dynamics_plan.csv")
    return {"status": "not evaluated", "reason": reason, "planned_experiments": len(planned), "files": files}
