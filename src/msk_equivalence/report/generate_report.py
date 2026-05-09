from __future__ import annotations

from pathlib import Path
from typing import Any

from msk_equivalence.utils import write_json, write_markdown


LEVELS = [
    ("Level 0", "视觉/结构等效", ["topology", "conventions"]),
    ("Level 1", "运动学等效", ["kinematics", "orientation_audit", "random_fk", "joint_sweep"]),
    ("Level 2", "肌骨几何等效", ["muscle_length", "moment_arm"]),
    ("Level 3", "动力学等效", ["inertial", "inverse_dynamics", "muscle_torque", "passive_forces"]),
    ("Level 3.5", "广义力矩分解", ["generalized_torque"]),
    ("Level 4 Native", "原生 OpenSim 任务行为等效", ["contact", "level4_correction_force_audit", "forward_dynamics"]),
    ("Level 4 Adapter", "OpenSim+runtime adapter 任务行为等效", ["forward_dynamics_adapter", "adapter_no_contact_rollout", "rl_distribution_rollout"]),
]


def _level_status(results: dict[str, dict[str, Any]], checks: list[str]) -> str:
    statuses = [results.get(name, {}).get("status", "not evaluated") for name in checks]
    if any(s == "failed" for s in statuses):
        return "failed"
    if any(s == "warning" for s in statuses):
        return "warning"
    if all(s == "passed" for s in statuses):
        return "passed"
    if any(s == "passed" for s in statuses):
        return "warning"
    return "not evaluated"


def _problem_lines(results: dict[str, dict[str, Any]]) -> list[str]:
    lines = []
    for name, result in results.items():
        status = result.get("status", "not evaluated")
        if status in {"failed", "warning", "not evaluated"}:
            reason = result.get("reason") or result.get("missing_mapped_entities") or result.get("sign_warning_count") or ""
            lines.append(f"- `{name}`: {status}. {reason}")
    return lines or ["- No major issues recorded by automated checks."]


def _verdict(levels: list[dict[str, Any]]) -> dict[str, str]:
    statuses = {item["level"]: item["status"] for item in levels}
    adapter_status = statuses.get("Level 4 Adapter")
    native_level4_status = statuses.get("Level 4 Native")
    core = [statuses.get("Level 0"), statuses.get("Level 1"), statuses.get("Level 2"), statuses.get("Level 3"), statuses.get("Level 3.5")]
    blocking_statuses = {
        key: value
        for key, value in statuses.items()
        if key != "Level 4 Native" and value == "failed"
    }
    if blocking_statuses:
        return {
            "status": "not equivalent",
            "reason": "At least one required core or adapter equivalence gate failed. Do not treat the models as interchangeable for RL training.",
        }
    if all(status == "passed" for status in core):
        if adapter_status == "passed":
            return {
                "status": "validated MuJoCo counterpart via explicit runtime adapter",
                "reason": "Core structural/geometric/dynamic gates passed and configured runtime-adapter behavior gates passed for the reported test distributions. Native OpenSim Level 4 remains a separately reported diagnostic.",
            }
        if native_level4_status == "passed":
            return {
                "status": "validated MuJoCo counterpart",
                "reason": "All configured structural, geometric, dynamic, generalized-torque and native task-behavior gates passed for the reported test distributions.",
            }
        return {
            "status": "dynamics equivalent; task behavior pending",
            "reason": "Core model and generalized-torque gates passed, but neither native nor runtime-adapter task-level behavior checks are fully passed.",
        }
    if statuses.get("Level 0") in {"passed", "warning"} and statuses.get("Level 1") == "passed":
        return {
            "status": "partially equivalent",
            "reason": "Rigid-body kinematics are usable, but muscle geometry, dynamics, or behavior gates still need work.",
        }
    return {
        "status": "not established",
        "reason": "Insufficient passed gates to claim equivalence.",
    }


def generate(out_dir: Path, results: dict[str, dict[str, Any]], inputs: dict[str, str]) -> dict[str, Any]:
    levels = []
    for code, title, checks in LEVELS:
        levels.append({"level": code, "title": title, "status": _level_status(results, checks), "checks": checks})
    verdict = _verdict(levels)
    summary = {"inputs": inputs, "verdict": verdict, "levels": levels, "checks": results}
    write_json(out_dir / "summary.json", summary)

    md = [
        "# OpenSim vs MuJoCo Musculoskeletal Equivalence Report",
        "",
        "## Inputs",
        "",
        f"- OpenSim: `{inputs.get('osim')}`",
        f"- MuJoCo: `{inputs.get('mjcf')}`",
        f"- Mapping: `{inputs.get('mapping')}`",
        f"- Adapter config: `{inputs.get('adapter_config') or 'not configured'}`",
        "",
        "## Equivalence Verdict",
        "",
        f"- status: `{verdict['status']}`",
        f"- reason: {verdict['reason']}",
        "",
        "## Layered Judgment",
        "",
        "| Level | Meaning | Status | Checks |",
        "| --- | --- | --- | --- |",
    ]
    for item in levels:
        md.append(f"| {item['level']} | {item['title']} | {item['status']} | {', '.join(item['checks'])} |")
    md.extend(["", "## Key Metrics", ""])
    for name, result in results.items():
        metrics = {k: v for k, v in result.items() if k not in {"files", "reason"}}
        md.append(f"### {name}")
        md.append("")
        for key, value in metrics.items():
            md.append(f"- {key}: `{value}`")
        files = result.get("files", [])
        if files:
            md.append(f"- files: {', '.join(f'`{f}`' for f in files)}")
        if result.get("reason"):
            md.append(f"- reason: {result['reason']}")
        md.append("")
    md.extend(["## Largest Problems And Next Fixes", ""])
    md.extend(_problem_lines(results))
    md.extend(
        [
            "",
            "## Recommendations",
            "",
            "- Resolve missing mapped bodies, coordinates, muscles and sites before interpreting numeric errors.",
            "- Validate coordinate signs with joint sweep and moment-arm sign consistency.",
            "- Treat inertia tensor comparison as provisional unless local body/inertial frames are aligned.",
            "- Use Level 2 muscle length and moment arm errors as the primary gate before RL muscle-control experiments.",
            "- Use Level 4 task behavior only after short inverse/forward dynamics checks are stable.",
            "- Treat native OpenSim Level 4 failure and runtime-adapter Level 4 results as separate claims; an adapter pass is not a pure `.osim` pass.",
            "- Do not claim arbitrary long-horizon bitwise identity; report the tested state/action distributions and known contact/integrator limitations.",
        ]
    )
    write_markdown(out_dir / "index.md", "\n".join(md) + "\n")
    return summary
