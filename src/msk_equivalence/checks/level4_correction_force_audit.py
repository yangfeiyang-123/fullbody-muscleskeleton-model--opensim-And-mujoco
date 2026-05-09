from __future__ import annotations

from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from msk_equivalence.checks.forward_dynamics import (
    _build_activation_adapter,
    _coordinate_force_probes,
    _independent_coordinate_items,
    _mujoco_forward,
    _mujoco_qacc_vector,
    _opensim_activation_values,
    _opensim_qacc_vector,
    _random_activation_vectors,
    _set_opensim_activations,
    _set_opensim_speeds,
)
from msk_equivalence.checks.kinematics import _pose_values
from msk_equivalence.checks.inverse_dynamics import _coupler_constraints, _with_coupled_coordinates
from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import rel_error, rmse, write_csv, write_markdown, write_worst_csv


def _force_expression(force: Any) -> str:
    try:
        return str(force.getExpression())
    except Exception:
        return ""


def _probe_expressions_from_xml(path: Path) -> dict[str, str]:
    expressions: dict[str, str] = {}
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return expressions
    for force in root.iter("ExpressionBasedCoordinateForce"):
        name = str(force.attrib.get("name", ""))
        if not name.startswith("level4_neutral_qacc_fit_"):
            continue
        coordinate = force.findtext("coordinate", default="").strip()
        expression = force.findtext("expression", default="").strip()
        if coordinate and expression:
            expressions[coordinate] = expression
    return expressions


def _set_probe_expressions(probes: dict[str, Any], expressions: dict[str, str]) -> None:
    for coordinate, force in probes.items():
        try:
            force.setExpression(str(expressions.get(coordinate, "0.0") or "0.0"))
        except Exception:
            continue


def _set_probe_enabled(probes: dict[str, Any], enabled: bool, original: dict[str, str]) -> None:
    if enabled:
        _set_probe_expressions(probes, original)
    else:
        _set_probe_expressions(probes, {coordinate: "0.0" for coordinate in probes})


def _neutral_pose(osim: Any, mapping: MappingConfig) -> tuple[dict[str, float], dict[str, float]]:
    sample = next((s for s in mapping.pose_samples if str(s.get("name", "")) == "neutral"), mapping.pose_samples[0])
    constraints = _coupler_constraints(osim)
    opensim_pose = _with_coupled_coordinates(_pose_values(sample, mapping, "opensim"), constraints, mapping, "opensim")
    mujoco_pose = _with_coupled_coordinates(_pose_values(sample, mapping, "mujoco"), constraints, mapping, "mujoco")
    return opensim_pose, mujoco_pose


def _realize_opensim_qacc(
    osim: Any,
    coordinate_items: list[dict[str, Any]],
    mapping: MappingConfig,
    pose: dict[str, float],
    activations: dict[str, float],
) -> np.ndarray:
    osim.state = osim.model.initSystem()
    osim.set_pose(pose)
    _set_opensim_speeds(osim, {})
    _set_opensim_activations(osim, activations)
    osim.model.realizeAcceleration(osim.state)
    return _opensim_qacc_vector(osim, coordinate_items, mapping)


def _qacc_rows(
    experiment: str,
    vector_index: int,
    coordinate_items: list[dict[str, Any]],
    mapping: MappingConfig,
    opensim_with_correction: np.ndarray,
    opensim_without_correction: np.ndarray,
    mujoco_qacc: np.ndarray,
    active_muscle_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_idx, item in enumerate(coordinate_items):
        oname = mapping.side_name(item, "opensim")
        mname = mapping.side_name(item, "mujoco")
        with_err = abs(float(opensim_with_correction[row_idx]) - float(mujoco_qacc[row_idx]))
        without_err = abs(float(opensim_without_correction[row_idx]) - float(mujoco_qacc[row_idx]))
        improvement = without_err - with_err
        rows.append(
            {
                "experiment": experiment,
                "vector_index": vector_index,
                "opensim_coordinate": oname,
                "mujoco_qvel": mname,
                "coordinate_role": "independent",
                "opensim_qacc_with_level4_correction": float(opensim_with_correction[row_idx]),
                "opensim_qacc_without_level4_correction": float(opensim_without_correction[row_idx]),
                "mujoco_qacc": float(mujoco_qacc[row_idx]),
                "absolute_error_with_correction": with_err,
                "absolute_error_without_correction": without_err,
                "correction_error_improvement": improvement,
                "relative_error_with_correction": rel_error(
                    with_err,
                    max(abs(float(opensim_with_correction[row_idx])), abs(float(mujoco_qacc[row_idx]))),
                ),
                "relative_error_without_correction": rel_error(
                    without_err,
                    max(abs(float(opensim_without_correction[row_idx])), abs(float(mujoco_qacc[row_idx]))),
                ),
                "active_muscle_count": active_muscle_count,
                "status": "evaluated",
                "diagnostic_only": True,
                "note": "Audits fitted OpenSim level4_neutral_qacc_fit_* coordinate forces on held-out acceleration comparisons.",
            }
        )
    return rows


def _random_vector_indices(mapping: MappingConfig) -> tuple[list[int], list[dict[str, float]]]:
    vectors = _random_activation_vectors(mapping)
    debug = mapping.raw.get("_debug", {})
    debug_vector_index = debug.get("vector_index") if isinstance(debug, dict) else None
    if debug_vector_index is None:
        return list(range(len(vectors))), vectors
    idx = int(debug_vector_index)
    if 0 <= idx < len(vectors):
        return [idx], [vectors[idx]]
    return [], []


def run(osim: Any, mjcf: Any, mapping: MappingConfig, out_dir: Path) -> dict[str, Any]:
    probes = _coordinate_force_probes(osim)
    if not probes:
        write_markdown(
            out_dir / "level4_correction_force_audit.md",
            "# Level 4 Correction Force Audit\n\nNo `level4_neutral_qacc_fit_*` probes were found.\n",
        )
        return {"status": "passed", "reason": "No fitted level4_neutral_qacc_fit_* correction probes were found.", "files": []}

    xml_expressions = _probe_expressions_from_xml(osim.path) if getattr(osim, "path", None) else {}
    original = {
        coordinate: xml_expressions.get(coordinate) or _force_expression(force) or "0.0"
        for coordinate, force in probes.items()
    }
    opensim_pose, mujoco_pose = _neutral_pose(osim, mapping)
    coordinate_items = _independent_coordinate_items(osim, mapping)
    adapter = _build_activation_adapter(osim, mjcf, mapping, opensim_pose, mujoco_pose)

    rows: list[dict[str, Any]] = []
    force_rows = [
        {
            "opensim_coordinate": coordinate,
            "force_name": str(force.getName()) if hasattr(force, "getName") else "",
            "original_expression": expression,
            "status": "evaluated",
        }
        for coordinate, force in sorted(probes.items())
        for expression in [original.get(coordinate, "")]
    ]

    try:
        _set_probe_enabled(probes, True, original)
        with_correction = _realize_opensim_qacc(osim, coordinate_items, mapping, opensim_pose, {})
        _set_probe_enabled(probes, False, original)
        without_correction = _realize_opensim_qacc(osim, coordinate_items, mapping, opensim_pose, {})
        _mujoco_forward(mjcf, mujoco_pose, {}, {}, disable_contact=True)
        mujoco_qacc = _mujoco_qacc_vector(mjcf, coordinate_items, mapping)
        rows.extend(
            _qacc_rows(
                "zero_activation_neutral",
                -1,
                coordinate_items,
                mapping,
                with_correction,
                without_correction,
                mujoco_qacc,
                0,
            )
        )

        vector_indices, vectors = _random_vector_indices(mapping)
        for vector_index, mujoco_activations in zip(vector_indices, vectors):
            opensim_activations = _opensim_activation_values(mujoco_activations, mapping, adapter)
            _set_probe_enabled(probes, True, original)
            with_correction = _realize_opensim_qacc(osim, coordinate_items, mapping, opensim_pose, opensim_activations)
            _set_probe_enabled(probes, False, original)
            without_correction = _realize_opensim_qacc(osim, coordinate_items, mapping, opensim_pose, opensim_activations)
            _mujoco_forward(mjcf, mujoco_pose, {}, mujoco_activations, disable_contact=True)
            mujoco_qacc = _mujoco_qacc_vector(mjcf, coordinate_items, mapping)
            rows.extend(
                _qacc_rows(
                    "sparse_random_activation_neutral",
                    vector_index,
                    coordinate_items,
                    mapping,
                    with_correction,
                    without_correction,
                    mujoco_qacc,
                    len(mujoco_activations),
                )
            )
    except Exception as exc:
        _set_probe_enabled(probes, True, original)
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}", "files": []}
    finally:
        _set_probe_enabled(probes, True, original)

    force_df = write_csv(out_dir / "diagnostics" / "level4_correction_force_probes.csv", force_rows)
    df = write_csv(out_dir / "level4_correction_force_qacc_audit.csv", rows)
    worst_with = write_worst_csv(out_dir / "diagnostics" / "level4_correction_force_worst_with_correction.csv", df, "absolute_error_with_correction")
    worst_without = write_worst_csv(
        out_dir / "diagnostics" / "level4_correction_force_worst_without_correction.csv",
        df,
        "absolute_error_without_correction",
    )
    finite = df[df["status"] == "evaluated"].copy() if not df.empty else df
    zero = finite[finite["experiment"] == "zero_activation_neutral"] if not finite.empty else finite
    random = finite[finite["experiment"] == "sparse_random_activation_neutral"] if not finite.empty else finite

    def metric(group: Any, column: str) -> tuple[float | None, float | None]:
        if group.empty:
            return None, None
        return float(group[column].max()), rmse(group[column])

    zero_with_max, zero_with_rmse = metric(zero, "absolute_error_with_correction")
    zero_without_max, zero_without_rmse = metric(zero, "absolute_error_without_correction")
    random_with_max, random_with_rmse = metric(random, "absolute_error_with_correction")
    random_without_max, random_without_rmse = metric(random, "absolute_error_without_correction")
    degraded = int((finite["correction_error_improvement"] < 0).sum()) if not finite.empty else 0
    improved = int((finite["correction_error_improvement"] > 0).sum()) if not finite.empty else 0
    fail_abs = float(mapping.thresholds.get("forward_dynamics_accel_fail", 20.0))
    fail_rel = float(mapping.thresholds.get("forward_dynamics_accel_fail_rel", 0.10))
    warn_abs = float(mapping.thresholds.get("forward_dynamics_accel_warning", 5.0))
    warn_rel = float(mapping.thresholds.get("forward_dynamics_accel_warning_rel", 0.05))
    random_rel = float(random["relative_error_with_correction"].max()) if not random.empty else None
    zero_rel = float(zero["relative_error_with_correction"].max()) if not zero.empty else None
    if random_with_max is None:
        status = "warning"
        reason = "No held-out random activation rows were evaluated for fitted correction forces."
    elif random_with_max > fail_abs or (random_rel is not None and random_rel > fail_rel):
        status = "failed"
        reason = (
            "Existing level4_neutral_qacc_fit_* forces reduce the neutral fit error but fail held-out random activation qacc. "
            "They must not be treated as native Level 4 validation."
        )
    elif zero_with_max is not None and (zero_with_max > fail_abs or (zero_rel is not None and zero_rel > fail_rel)):
        status = "failed"
        reason = "Existing level4_neutral_qacc_fit_* forces fail zero-activation qacc under the configured Level 4 gate."
    elif random_with_max > warn_abs or (random_rel is not None and random_rel > warn_rel):
        status = "warning"
        reason = "Existing level4_neutral_qacc_fit_* forces are below failure thresholds but above warning thresholds on held-out random qacc."
    else:
        status = "passed"
        reason = "Existing level4_neutral_qacc_fit_* forces did not fail the held-out qacc audit; rollout gates remain separate."
    files = [
        "level4_correction_force_qacc_audit.csv",
        "level4_correction_force_audit.md",
        "diagnostics/level4_correction_force_probes.csv",
    ]
    if worst_with:
        files.append(f"diagnostics/{worst_with}")
    if worst_without:
        files.append(f"diagnostics/{worst_without}")

    write_markdown(
        out_dir / "level4_correction_force_audit.md",
        "\n".join(
            [
                "# Level 4 Correction Force Audit",
                "",
                "This diagnostic evaluates the existing `level4_neutral_qacc_fit_*` OpenSim coordinate forces.",
                "These forces are fitted correction probes, not a native MuJoCo counterpart mechanism.",
                "",
                f"- probe_count: `{len(force_df)}`",
                f"- evaluated_rows: `{len(finite) if not finite.empty else 0}`",
                f"- improved_rows: `{improved}`",
                f"- degraded_rows: `{degraded}`",
                f"- zero_activation_max_abs_error_with_correction: `{zero_with_max}`",
                f"- zero_activation_rmse_with_correction: `{zero_with_rmse}`",
                f"- zero_activation_max_abs_error_without_correction: `{zero_without_max}`",
                f"- zero_activation_rmse_without_correction: `{zero_without_rmse}`",
                f"- random_activation_max_abs_error_with_correction: `{random_with_max}`",
                f"- random_activation_rmse_with_correction: `{random_with_rmse}`",
                f"- random_activation_max_relative_error_with_correction: `{random_rel}`",
                f"- random_activation_max_abs_error_without_correction: `{random_without_max}`",
                f"- random_activation_rmse_without_correction: `{random_without_rmse}`",
                f"- status: `{status}`",
                "",
                reason,
            ]
        )
        + "\n",
    )
    return {
        "status": status,
        "reason": reason,
        "probe_count": int(len(force_df)),
        "evaluated_rows": int(len(finite)) if not finite.empty else 0,
        "improved_rows": improved,
        "degraded_rows": degraded,
        "zero_activation_max_abs_error_with_correction": zero_with_max,
        "zero_activation_rmse_with_correction": zero_with_rmse,
        "zero_activation_max_abs_error_without_correction": zero_without_max,
        "zero_activation_rmse_without_correction": zero_without_rmse,
        "random_activation_max_abs_error_with_correction": random_with_max,
        "random_activation_rmse_with_correction": random_with_rmse,
        "random_activation_max_relative_error_with_correction": random_rel,
        "random_activation_max_abs_error_without_correction": random_without_max,
        "random_activation_rmse_without_correction": random_without_rmse,
        "files": files,
    }
