from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from msk_equivalence.checks.forward_dynamics import (
    _build_activation_adapter,
    _independent_coordinate_items,
    _mujoco_forward,
    _opensim_activation_values,
    _qvel_index,
    _random_activation_vectors,
    _set_opensim_activations,
    _set_opensim_speeds,
)
from msk_equivalence.checks.inverse_dynamics import _coupler_constraints, _with_coupled_coordinates
from msk_equivalence.checks.kinematics import _pose_values
from msk_equivalence.checks.moment_arm import _pairs
from msk_equivalence.checks.muscle_torque import _mujoco_actuator_force, _opensim_actuation
from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import rel_error, rmse, write_csv, write_markdown, write_worst_csv


def _debug(mapping: MappingConfig) -> dict[str, Any]:
    value = mapping.raw.get("_debug", {})
    return value if isinstance(value, dict) else {}


def _sample_poses(osim: Any, mapping: MappingConfig) -> tuple[dict[str, float], dict[str, float]]:
    sample = next((s for s in mapping.pose_samples if str(s.get("name", "")) == "neutral"), mapping.pose_samples[0])
    constraints = _coupler_constraints(osim)
    opensim_pose = _with_coupled_coordinates(_pose_values(sample, mapping, "opensim"), constraints, mapping, "opensim")
    mujoco_pose = _with_coupled_coordinates(_pose_values(sample, mapping, "mujoco"), constraints, mapping, "mujoco")
    return opensim_pose, mujoco_pose


def _coordinate_sign(item: dict[str, Any]) -> float:
    return -1.0 if bool(item.get("sign_flip", False)) else 1.0


def _active_force(
    osim: Any,
    mjcf: Any,
    opensim_muscle: str,
    mujoco_actuator: str,
    opensim_activation: float,
    mujoco_activation: float,
    opensim_pose: dict[str, float],
    mujoco_pose: dict[str, float],
) -> tuple[float, float]:
    opensim_baseline = abs(_opensim_actuation(osim, opensim_muscle, 0.0, opensim_pose))
    mujoco_baseline = _mujoco_actuator_force(mjcf, mujoco_actuator, 0.0, mujoco_pose)
    opensim_force = abs(_opensim_actuation(osim, opensim_muscle, opensim_activation, opensim_pose)) - opensim_baseline
    mujoco_force = _mujoco_actuator_force(mjcf, mujoco_actuator, mujoco_activation, mujoco_pose) - mujoco_baseline
    return float(opensim_force), float(mujoco_force)


def _mujoco_mass_diag(mjcf: Any) -> np.ndarray:
    mj = mjcf.mujoco
    dense = np.zeros((int(mjcf.model.nv), int(mjcf.model.nv)), dtype=float)
    mj.mj_fullM(mjcf.model, dense, mjcf.data.qM)
    return np.diag(dense)


def _coupled_opensim_moment_arm(osim: Any, muscle_name: str, coordinate_name: str, base_pose: dict[str, float], constraints: list[dict[str, Any]], mapping: MappingConfig, eps: float) -> float:
    plus = dict(base_pose)
    minus = dict(base_pose)
    base = float(base_pose.get(coordinate_name, osim.default_coordinates.get(coordinate_name, 0.0)))
    plus[coordinate_name] = base + eps
    minus[coordinate_name] = base - eps
    plus = _with_coupled_coordinates(plus, constraints, mapping, "opensim")
    minus = _with_coupled_coordinates(minus, constraints, mapping, "opensim")
    osim.set_pose(plus)
    lp = float(osim.muscle_length(muscle_name))
    osim.set_pose(minus)
    lm = float(osim.muscle_length(muscle_name))
    osim.set_pose(base_pose)
    return -float((lp - lm) / (2.0 * eps))


def _coupled_mujoco_moment_arm(mjcf: Any, tendon_name: str, coordinate_name: str, base_pose: dict[str, float], constraints: list[dict[str, Any]], mapping: MappingConfig, eps: float) -> float:
    plus = dict(base_pose)
    minus = dict(base_pose)
    base = float(base_pose.get(coordinate_name, 0.0))
    plus[coordinate_name] = base + eps
    minus[coordinate_name] = base - eps
    plus = _with_coupled_coordinates(plus, constraints, mapping, "mujoco")
    minus = _with_coupled_coordinates(minus, constraints, mapping, "mujoco")
    mjcf.set_pose(plus)
    lp = float(mjcf.tendon_length(tendon_name))
    mjcf.set_pose(minus)
    lm = float(mjcf.tendon_length(tendon_name))
    mjcf.set_pose(base_pose)
    return -float((lp - lm) / (2.0 * eps))


def _qacc_by_coordinate(osim: Any, mjcf: Any, mapping: MappingConfig, coordinate_items: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    coord_set = osim.model.getCoordinateSet()
    mass_diag = _mujoco_mass_diag(mjcf)
    for item in coordinate_items:
        oname = mapping.side_name(item, "opensim")
        mname = mapping.side_name(item, "mujoco")
        if not oname or not mname:
            continue
        try:
            qvel_idx = _qvel_index(mjcf, mname)
            if qvel_idx is None:
                continue
            sign = _coordinate_sign(item)
            opensim_qacc = float(coord_set.get(oname).getAccelerationValue(osim.state))
            mujoco_qacc = sign * float(mjcf.data.qacc[qvel_idx])
            rows[oname] = {
                "opensim_qacc": opensim_qacc,
                "mujoco_qacc": mujoco_qacc,
                "qacc_error": abs(opensim_qacc - mujoco_qacc),
                "mujoco_mass_diag": float(mass_diag[qvel_idx]) if qvel_idx < len(mass_diag) else np.nan,
            }
        except Exception:
            continue
    return rows


def _contribution_rows(osim: Any, mjcf: Any, mapping: MappingConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    opensim_pose, mujoco_pose = _sample_poses(osim, mapping)
    constraints = _coupler_constraints(osim)
    coordinate_items = _independent_coordinate_items(osim, mapping)
    coord_by_name = {mapping.side_name(item, "opensim"): item for item in coordinate_items}
    adapter = _build_activation_adapter(osim, mjcf, mapping, opensim_pose, mujoco_pose)
    pairs = [
        (muscle, coord)
        for muscle, coord in _pairs(mapping)
        if mapping.side_name(coord, "opensim") in coord_by_name
    ]
    configured_pairs = {
        (mapping.side_name(muscle, "opensim"), mapping.side_name(coord, "opensim"))
        for muscle, coord in pairs
    }
    dbg = _debug(mapping)
    debug_coordinate = dbg.get("coordinate")
    debug_vector_index = dbg.get("vector_index")
    debug_muscle = dbg.get("muscle")

    contributions: list[dict[str, Any]] = []
    vector_summaries: list[dict[str, Any]] = []
    active_rows: list[dict[str, Any]] = []
    amplification_rows: list[dict[str, Any]] = []
    missing_coverage_rows: list[dict[str, Any]] = []
    coverage_moment_arm_cache: dict[tuple[str, str], tuple[float, float, str]] = {}
    vectors = _random_activation_vectors(mapping)
    if debug_vector_index is not None:
        vectors = [vectors[int(debug_vector_index)]] if 0 <= int(debug_vector_index) < len(vectors) else []
        vector_indices = [int(debug_vector_index)] if vectors else []
    else:
        vector_indices = list(range(len(vectors)))

    for vector_index, mujoco_activations in zip(vector_indices, vectors):
        if debug_muscle:
            mapped_mujoco = {
                mapping.side_name(item, "opensim"): mapping.side_name(item, "mujoco")
                for item in mapping.muscles
            }.get(str(debug_muscle))
            mujoco_activations = {mapped_mujoco: mujoco_activations.get(mapped_mujoco, 1.0)} if mapped_mujoco else {}
        opensim_activations = _opensim_activation_values(mujoco_activations, mapping, adapter)

        osim.set_pose(opensim_pose)
        _set_opensim_speeds(osim, {})
        _set_opensim_activations(osim, opensim_activations)
        try:
            osim.model.realizeAcceleration(osim.state)
        except Exception:
            pass
        _mujoco_forward(mjcf, mujoco_pose, {}, mujoco_activations, disable_contact=True)
        qacc = _qacc_by_coordinate(osim, mjcf, mapping, coordinate_items)

        totals: dict[str, dict[str, float]] = defaultdict(lambda: {"opensim_tau": 0.0, "mujoco_tau": 0.0, "max_force_error": 0.0, "max_moment_arm_error": 0.0, "contributors": 0.0})
        active_rows.append(
            {
                "vector_index": vector_index,
                "active_muscle_count": len(mujoco_activations),
                "active_mujoco_actuators": " ".join(sorted(mujoco_activations)),
                "mapped_opensim_muscles": " ".join(
                    sorted(
                        mapping.side_name(item, "opensim") or ""
                        for item in mapping.muscles
                        if mapping.side_name(item, "mujoco") in mujoco_activations
                    )
                ).strip(),
            }
        )

        force_cache: dict[str, tuple[float, float]] = {}
        for muscle_item, coord_item in pairs:
            oname = mapping.side_name(muscle_item, "opensim")
            mname = mapping.side_name(muscle_item, "mujoco")
            oc = mapping.side_name(coord_item, "opensim")
            mc = mapping.side_name(coord_item, "mujoco")
            if not oname or not mname or not oc or not mc:
                continue
            if debug_coordinate and oc != debug_coordinate:
                continue
            if debug_muscle and oname != debug_muscle:
                continue
            mujoco_activation = float(mujoco_activations.get(mname, 0.0))
            opensim_activation = float(opensim_activations.get(oname, 0.0))
            if mujoco_activation <= 0.0 and opensim_activation <= 0.0:
                continue
            try:
                if oname not in force_cache:
                    force_cache[oname] = _active_force(osim, mjcf, oname, mname, opensim_activation, mujoco_activation, opensim_pose, mujoco_pose)
                opensim_force, mujoco_force = force_cache[oname]
                eps = float(mapping.thresholds.get("moment_arm_fd_epsilon", 1e-6))
                opensim_ma = _coupled_opensim_moment_arm(osim, oname, oc, opensim_pose, constraints, mapping, eps)
                mujoco_ma = _coordinate_sign(coord_item) * _coupled_mujoco_moment_arm(mjcf, mname, mc, mujoco_pose, constraints, mapping, eps)
                projected_coordinate = oc
                projection_derivative = 1.0
                opensim_tau = opensim_force * opensim_ma
                mujoco_tau = mujoco_force * mujoco_ma
                projected_opensim_tau = projection_derivative * opensim_tau
                projected_mujoco_tau = projection_derivative * mujoco_tau
                tau_error = abs(projected_opensim_tau - projected_mujoco_tau)
                force_error = abs(opensim_force - mujoco_force)
                ma_error = abs(opensim_ma - mujoco_ma)
                status = "evaluated"
            except Exception as exc:
                projected_coordinate = ""
                projection_derivative = np.nan
                opensim_force = mujoco_force = opensim_ma = mujoco_ma = opensim_tau = mujoco_tau = projected_opensim_tau = projected_mujoco_tau = tau_error = force_error = ma_error = np.nan
                status = f"skipped: {exc}"
            if status == "evaluated":
                total = totals[oc]
                total["opensim_tau"] += float(projected_opensim_tau)
                total["mujoco_tau"] += float(projected_mujoco_tau)
                total["max_force_error"] = max(total["max_force_error"], float(force_error))
                total["max_moment_arm_error"] = max(total["max_moment_arm_error"], float(ma_error))
                total["contributors"] += 1.0
            contributions.append(
                {
                    "experiment": "sparse_random_activation_generalized_torque",
                    "vector_index": vector_index,
                    "opensim_coordinate": oc,
                    "mujoco_coordinate": mc,
                    "projected_opensim_coordinate": projected_coordinate,
                    "constraint_projection_derivative": projection_derivative,
                    "opensim_muscle": oname,
                    "mujoco_actuator": mname,
                    "mujoco_activation": mujoco_activation,
                    "opensim_adapter_activation": opensim_activation,
                    "opensim_active_force_n": opensim_force,
                    "mujoco_active_force_n": mujoco_force,
                    "force_error_n": force_error,
                    "opensim_moment_arm_m": opensim_ma,
                    "mujoco_moment_arm_m": mujoco_ma,
                    "moment_arm_error_m": ma_error,
                    "opensim_tau_nm": opensim_tau,
                    "mujoco_tau_nm": mujoco_tau,
                    "projected_opensim_tau_nm": projected_opensim_tau,
                    "projected_mujoco_tau_nm": projected_mujoco_tau,
                    "absolute_torque_error_nm": tau_error,
                    "status": status,
                }
            )

        for oc, total in totals.items():
            torque_error = abs(total["opensim_tau"] - total["mujoco_tau"])
            qacc_row = qacc.get(oc, {})
            qacc_error = float(qacc_row.get("qacc_error", np.nan))
            amplification = qacc_error / torque_error if np.isfinite(qacc_error) and torque_error > 1e-12 else np.nan
            sign_mismatch = (
                abs(total["opensim_tau"]) > 1e-9
                and abs(total["mujoco_tau"]) > 1e-9
                and np.sign(total["opensim_tau"]) != np.sign(total["mujoco_tau"])
            )
            if sign_mismatch:
                likely = "torque sign mismatch"
            elif total["max_moment_arm_error"] > float(mapping.thresholds.get("moment_arm_independent_fail_m", 0.1)):
                likely = "moment arm mismatch"
            elif total["max_force_error"] > float(mapping.thresholds.get("muscle_force_fail_n", 3.0)):
                likely = "muscle force mismatch"
            elif np.isfinite(amplification) and amplification > 100.0:
                likely = "mass matrix / small-inertia amplification"
            else:
                likely = "torque mismatch or passive/bias contribution"
            vector_summaries.append(
                {
                    "experiment": "sparse_random_activation_generalized_torque",
                    "vector_index": vector_index,
                    "opensim_coordinate": oc,
                    "contributors": int(total["contributors"]),
                    "opensim_total_tau_nm": total["opensim_tau"],
                    "mujoco_total_tau_nm": total["mujoco_tau"],
                    "absolute_torque_error_nm": torque_error,
                    "relative_torque_error": rel_error(torque_error, max(abs(total["opensim_tau"]), abs(total["mujoco_tau"]))),
                    "max_force_error_n": total["max_force_error"],
                    "max_moment_arm_error_m": total["max_moment_arm_error"],
                    "opensim_qacc": qacc_row.get("opensim_qacc", np.nan),
                    "mujoco_qacc": qacc_row.get("mujoco_qacc", np.nan),
                    "qacc_error": qacc_error,
                    "mujoco_mass_diag": qacc_row.get("mujoco_mass_diag", np.nan),
                    "qacc_error_per_torque_error": amplification,
                    "likely_failure_source": likely,
                    "status": "evaluated",
                }
            )
            amplification_rows.append(
                {
                    "vector_index": vector_index,
                    "opensim_coordinate": oc,
                    "absolute_torque_error_nm": torque_error,
                    "qacc_error": qacc_error,
                    "mujoco_mass_diag": qacc_row.get("mujoco_mass_diag", np.nan),
                    "qacc_error_per_torque_error": amplification,
                    "likely_failure_source": likely,
                }
            )
        for item in coordinate_items:
            oc = mapping.side_name(item, "opensim")
            mc = mapping.side_name(item, "mujoco")
            if not oc or oc in totals:
                continue
            if debug_coordinate and oc != debug_coordinate:
                continue
            qacc_row = qacc.get(oc, {})
            qacc_error = float(qacc_row.get("qacc_error", np.nan))
            if not np.isfinite(qacc_error):
                continue
            missing_coverage_qacc_fail = float(mapping.thresholds.get("generalized_torque_missing_coverage_fail_qacc", 100.0))
            missing_coverage_ma_threshold = float(mapping.thresholds.get("generalized_torque_missing_coverage_moment_arm_threshold_m", 1e-5))
            if qacc_error > missing_coverage_qacc_fail and mc:
                for muscle_item in mapping.muscles:
                    oname = mapping.side_name(muscle_item, "opensim")
                    mname = mapping.side_name(muscle_item, "mujoco")
                    if not oname or not mname or mname not in mujoco_activations:
                        continue
                    if (oname, oc) in configured_pairs:
                        continue
                    cache_key = (oname, oc)
                    if cache_key not in coverage_moment_arm_cache:
                        try:
                            eps = float(mapping.thresholds.get("moment_arm_fd_epsilon", 1e-6))
                            opensim_ma = _coupled_opensim_moment_arm(osim, oname, oc, opensim_pose, constraints, mapping, eps)
                            mujoco_ma = _coordinate_sign(item) * _coupled_mujoco_moment_arm(mjcf, mname, mc, mujoco_pose, constraints, mapping, eps)
                            coverage_status = "evaluated"
                        except Exception as exc:
                            opensim_ma = np.nan
                            mujoco_ma = np.nan
                            coverage_status = f"skipped: {exc}"
                        coverage_moment_arm_cache[cache_key] = (float(opensim_ma), float(mujoco_ma), coverage_status)
                    opensim_ma, mujoco_ma, coverage_status = coverage_moment_arm_cache[cache_key]
                    if coverage_status != "evaluated":
                        continue
                    max_abs_ma = max(abs(opensim_ma), abs(mujoco_ma))
                    if max_abs_ma <= missing_coverage_ma_threshold:
                        continue
                    missing_coverage_rows.append(
                        {
                            "experiment": "sparse_random_activation_generalized_torque",
                            "vector_index": vector_index,
                            "opensim_coordinate": oc,
                            "mujoco_coordinate": mc,
                            "opensim_muscle": oname,
                            "mujoco_actuator": mname,
                            "mujoco_activation": float(mujoco_activations.get(mname, 0.0)),
                            "opensim_moment_arm_m": opensim_ma,
                            "mujoco_moment_arm_m": mujoco_ma,
                            "max_abs_moment_arm_m": max_abs_ma,
                            "qacc_error": qacc_error,
                            "qacc_fail_threshold": missing_coverage_qacc_fail,
                            "configured_pair_present": False,
                            "suggested_mapping": f"- muscle: {oname}; coordinate: {oc}",
                            "status": "failed: active muscle-coordinate moment arm is not covered by moment_arm_pairs",
                        }
                    )
            likely = "passive/damping/limit/bias contribution or coordinate coupling"
            vector_summaries.append(
                {
                    "experiment": "sparse_random_activation_generalized_torque",
                    "vector_index": vector_index,
                    "opensim_coordinate": oc,
                    "contributors": 0,
                    "opensim_total_tau_nm": 0.0,
                    "mujoco_total_tau_nm": 0.0,
                    "absolute_torque_error_nm": 0.0,
                    "relative_torque_error": np.nan,
                    "max_force_error_n": 0.0,
                    "max_moment_arm_error_m": 0.0,
                    "opensim_qacc": qacc_row.get("opensim_qacc", np.nan),
                    "mujoco_qacc": qacc_row.get("mujoco_qacc", np.nan),
                    "qacc_error": qacc_error,
                    "mujoco_mass_diag": qacc_row.get("mujoco_mass_diag", np.nan),
                    "qacc_error_per_torque_error": np.nan,
                    "likely_failure_source": likely,
                    "status": "diagnostic: no active mapped muscle contribution",
                }
            )
            amplification_rows.append(
                {
                    "vector_index": vector_index,
                    "opensim_coordinate": oc,
                    "absolute_torque_error_nm": 0.0,
                    "qacc_error": qacc_error,
                    "mujoco_mass_diag": qacc_row.get("mujoco_mass_diag", np.nan),
                    "qacc_error_per_torque_error": np.nan,
                    "likely_failure_source": likely,
                }
            )
    return contributions, vector_summaries, amplification_rows, active_rows, missing_coverage_rows


def _status(summary_df: Any, mapping: MappingConfig, missing_coverage_df: Any | None = None) -> str:
    if missing_coverage_df is not None and not missing_coverage_df.empty:
        return "failed"
    if summary_df.empty:
        return "not evaluated"
    finite = summary_df[summary_df["status"] == "evaluated"].copy()
    if finite.empty:
        return "not evaluated"
    abs_err = np.array(finite["absolute_torque_error_nm"], dtype=float)
    rel_err = np.array(finite["relative_torque_error"], dtype=float)
    reference = np.maximum(np.abs(np.array(finite["opensim_total_tau_nm"], dtype=float)), np.abs(np.array(finite["mujoco_total_tau_nm"], dtype=float)))
    warn_abs = float(mapping.thresholds.get("generalized_torque_warning_nm", mapping.thresholds.get("inverse_dynamics_warning_nm", 1.0)))
    fail_abs = float(mapping.thresholds.get("generalized_torque_fail_nm", mapping.thresholds.get("inverse_dynamics_fail_nm", 3.0)))
    warn_rel = float(mapping.thresholds.get("generalized_torque_warning_rel", 0.05))
    fail_rel = float(mapping.thresholds.get("generalized_torque_fail_rel", 0.10))
    rel_floor = float(mapping.thresholds.get("generalized_torque_relative_reference_floor_nm", warn_abs))
    rel_abs_floor = float(mapping.thresholds.get("generalized_torque_relative_abs_error_floor_nm", 0.1))
    finite_rel = rel_err[np.isfinite(rel_err) & (reference >= rel_floor) & (abs_err >= rel_abs_floor)]
    if np.nanmax(abs_err) > fail_abs or (finite_rel.size and np.nanmax(finite_rel) > fail_rel):
        return "failed"
    if np.nanmax(abs_err) > warn_abs or (finite_rel.size and np.nanmax(finite_rel) > warn_rel):
        return "warning"
    return "passed"


def run(osim: Any, mjcf: Any, mapping: MappingConfig, out_dir: Path) -> dict[str, Any]:
    contributions, summaries, amplification, active, missing_coverage = _contribution_rows(osim, mjcf, mapping)
    contribution_df = write_csv(out_dir / "diagnostics" / "per_muscle_contribution_worst_rows.csv", contributions)
    summary_df = write_csv(out_dir / "diagnostics" / "random_activation_torque_errors.csv", summaries)
    amplification_df = write_csv(out_dir / "diagnostics" / "inertia_amplification_report.csv", amplification)
    missing_coverage_df = write_csv(out_dir / "diagnostics" / "generalized_torque_missing_coverage.csv", missing_coverage)
    write_csv(out_dir / "diagnostics" / "active_muscles_in_failing_vectors.csv", active)
    write_csv(out_dir / "diagnostics" / "per_coordinate_torque_decomposition.csv", summaries)
    write_markdown(
        out_dir / "generalized_torque_report.md",
        "# Generalized Torque Decomposition\n\n"
        "This Level 3.5 gate compares active-muscle generalized torque at the matched neutral no-contact pose using the same sparse random activation vectors as Level 4. "
        "Each torque contribution is reconstructed as active muscle force times a constraint-chain finite-difference moment arm for directly comparable independent coordinates. "
        "Dependent-coordinate direct moment arms are not projected into the gate because OpenSim `computeMomentArm` for constrained dependent coordinates is not a plain partial derivative; finite-difference total derivatives along the coupled coordinate path are the comparable quantity. "
        "MuJoCo remains the reference; OpenSim activations use the existing force adapter that accounts for Thelen2003Muscle activation semantics.\n\n"
        "The report separates force mismatch, moment-arm mismatch, torque sign mismatch, missing active muscle-coordinate coverage, and small-inertia qacc amplification. "
        "It is a diagnostic bridge between Level 3 force/moment-arm checks and Level 4 forward dynamics; it does not claim arbitrary long-horizon bitwise identity.\n",
    )
    files = [
        "generalized_torque_report.md",
        "diagnostics/random_activation_torque_errors.csv",
        "diagnostics/per_coordinate_torque_decomposition.csv",
        "diagnostics/per_muscle_contribution_worst_rows.csv",
        "diagnostics/inertia_amplification_report.csv",
        "diagnostics/generalized_torque_missing_coverage.csv",
        "diagnostics/active_muscles_in_failing_vectors.csv",
    ]
    worst = write_worst_csv(out_dir / "diagnostics" / "generalized_torque_worst_error.csv", summary_df, "absolute_torque_error_nm")
    if worst:
        files.append(f"diagnostics/{worst}")
    worst_contrib = write_worst_csv(out_dir / "diagnostics" / "per_muscle_contribution_worst_torque_error.csv", contribution_df, "absolute_torque_error_nm")
    if worst_contrib:
        files.append(f"diagnostics/{worst_contrib}")
    worst_amp = write_worst_csv(out_dir / "diagnostics" / "inertia_amplification_worst.csv", amplification_df, "qacc_error_per_torque_error")
    if worst_amp:
        files.append(f"diagnostics/{worst_amp}")
    worst_qacc = write_worst_csv(out_dir / "diagnostics" / "generalized_torque_qacc_worst_error.csv", summary_df, "qacc_error")
    if worst_qacc:
        files.append(f"diagnostics/{worst_qacc}")
    worst_missing_coverage = write_worst_csv(out_dir / "diagnostics" / "generalized_torque_missing_coverage_worst.csv", missing_coverage_df, "qacc_error")
    if worst_missing_coverage:
        files.append(f"diagnostics/{worst_missing_coverage}")
    finite = summary_df[summary_df["status"] == "evaluated"] if not summary_df.empty else summary_df
    status = _status(summary_df, mapping, missing_coverage_df)
    reason = None
    if status == "failed":
        if not missing_coverage_df.empty:
            reason = "Random activation generalized-torque coverage is incomplete; inspect diagnostics/generalized_torque_missing_coverage.csv before trusting torque totals."
        else:
            reason = "Random activation generalized-torque mismatch exceeds thresholds; inspect diagnostics/generalized_torque_worst_error.csv and per-muscle contribution diagnostics."
    return {
        "status": status,
        **({"reason": reason} if reason else {}),
        "evaluated_rows": int(len(finite)) if not finite.empty else 0,
        "max_abs_torque_error_nm": float(finite["absolute_torque_error_nm"].max()) if not finite.empty else None,
        "rmse_abs_torque_error_nm": rmse(finite["absolute_torque_error_nm"]) if not finite.empty else None,
        "max_qacc_error_seen": float(finite["qacc_error"].max()) if not finite.empty else None,
        "max_qacc_error_per_torque_error": float(finite["qacc_error_per_torque_error"].replace([np.inf, -np.inf], np.nan).max()) if not finite.empty else None,
        "missing_coverage_rows": int(len(missing_coverage_df)) if not missing_coverage_df.empty else 0,
        "missing_coverage_max_qacc_error": float(missing_coverage_df["qacc_error"].max()) if not missing_coverage_df.empty else None,
        "missing_coverage_moment_arm_threshold_m": float(mapping.thresholds.get("generalized_torque_missing_coverage_moment_arm_threshold_m", 1e-5)),
        "missing_coverage_qacc_fail_threshold": float(mapping.thresholds.get("generalized_torque_missing_coverage_fail_qacc", 100.0)),
        "warning_threshold_nm": float(mapping.thresholds.get("generalized_torque_warning_nm", mapping.thresholds.get("inverse_dynamics_warning_nm", 1.0))),
        "failure_threshold_nm": float(mapping.thresholds.get("generalized_torque_fail_nm", mapping.thresholds.get("inverse_dynamics_fail_nm", 3.0))),
        "relative_error_abs_floor_nm": float(mapping.thresholds.get("generalized_torque_relative_abs_error_floor_nm", 0.1)),
        "note": "Level 3.5 active generalized-torque decomposition for sparse random activation vectors. Contact and long-horizon RL rollouts are separate claims.",
        "files": files,
    }
