from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path
from typing import Any

from msk_equivalence.adapters.runtime_dynamics import load_adapter_config
from msk_equivalence.errors import DependencyMissingError, ModelLoadError
from msk_equivalence.loaders.mujoco_loader import MuJoCoModel
from msk_equivalence.loaders.opensim_loader import OpenSimModel
from msk_equivalence.mapping import MappingConfig
from msk_equivalence.report.generate_report import generate
from msk_equivalence.utils import ensure_dir, setup_logging


DEFAULT_CHECKS = [
    "topology",
    "conventions",
    "inertial",
    "kinematics",
    "orientation_audit",
    "random_fk",
    "joint_sweep",
    "muscle_length",
    "moment_arm",
    "muscle_parameters",
    "muscle_torque",
    "passive_forces",
    "generalized_torque",
    "contact",
    "level4_correction_force_audit",
    "inverse_dynamics",
    "forward_dynamics",
    "forward_dynamics_adapter",
    "adapter_no_contact_rollout",
    "rl_distribution_rollout",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare OpenSim and MuJoCo full-body musculoskeletal model equivalence.")
    parser.add_argument("--osim", required=True, type=Path, help="Path to the OpenSim .osim model.")
    parser.add_argument("--mjcf", required=True, type=Path, help="Path to the MuJoCo .xml/.mjcf model.")
    parser.add_argument("--mapping", required=True, type=Path, help="Path to model_mapping.yaml.")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for the report.")
    parser.add_argument("--adapter-config", type=Path, help="Optional runtime dynamics adapter validation config.")
    parser.add_argument("--checks", nargs="*", default=DEFAULT_CHECKS, help="Check module names to run. Defaults to all checks.")
    parser.add_argument("--debug-coordinate", help="Restrict supported diagnostic checks to one OpenSim coordinate.")
    parser.add_argument("--debug-vector-index", type=int, help="Restrict supported diagnostic checks to one random activation vector.")
    parser.add_argument("--debug-muscle", help="Restrict supported diagnostic checks to one OpenSim muscle.")
    parser.add_argument("--fail-on-gate", action="store_true", help="Return exit code 1 when the equivalence verdict is not equivalent.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def run_check(name: str, osim: Any, mjcf: Any, mapping: MappingConfig, out_dir: Path, logger: logging.Logger) -> dict[str, Any]:
    logger.info("Running check: %s", name)
    try:
        module = importlib.import_module(f"msk_equivalence.checks.{name}")
        return module.run(osim, mjcf, mapping, out_dir)
    except Exception as exc:
        logger.exception("Check %s failed", name)
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}", "files": []}


def main() -> int:
    args = _parse_args()
    logger = setup_logging(args.verbose)
    out_dir = ensure_dir(args.out)
    try:
        mapping = MappingConfig.load(args.mapping)
        mapping.raw["_debug"] = {
            "coordinate": args.debug_coordinate,
            "vector_index": args.debug_vector_index,
            "muscle": args.debug_muscle,
        }
        mapping.raw["_adapter_config"] = load_adapter_config(args.adapter_config)
        osim = OpenSimModel.load(args.osim)
        mjcf = MuJoCoModel.load(args.mjcf)
    except (DependencyMissingError, ModelLoadError, FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("%s", exc)
        return 2

    results: dict[str, dict[str, Any]] = {}
    for check in args.checks:
        results[check] = run_check(check, osim, mjcf, mapping, out_dir, logger)
    summary = generate(
        out_dir,
        results,
        {
            "osim": str(args.osim),
            "mjcf": str(args.mjcf),
            "mapping": str(args.mapping),
            "adapter_config": str(args.adapter_config) if args.adapter_config else "",
            "out": str(args.out),
        },
    )
    logger.info("Report written to %s", out_dir / "index.md")
    passing_statuses = {
        "equivalent",
        "validated MuJoCo counterpart",
        "validated MuJoCo counterpart via explicit runtime adapter",
    }
    if args.fail_on_gate and summary.get("verdict", {}).get("status") not in passing_statuses:
        logger.error("Equivalence gate did not pass: %s", summary.get("verdict", {}).get("status"))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
