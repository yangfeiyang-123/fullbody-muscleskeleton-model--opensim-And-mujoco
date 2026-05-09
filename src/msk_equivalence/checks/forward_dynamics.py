from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from msk_equivalence.checks.kinematics import _pose_values
from msk_equivalence.checks.inverse_dynamics import _coupler_constraints, _dependent_coordinates, _with_coupled_coordinates
from msk_equivalence.checks.muscle_torque import _calibration_values, _interp_activation_for_force, _mujoco_actuator_force, _opensim_actuation
from msk_equivalence.loaders.opensim_loader import OpenSimModel
from msk_equivalence.mapping import MappingConfig
from msk_equivalence.utils import rel_error, rmse, write_csv, write_markdown, write_worst_csv


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
                "input_profile": item.get("input_profile", ""),
                "activation_profile": item.get("activation_profile", ""),
                "contact_probe_grid": item.get("contact_probe_grid", ""),
                "torque_coordinates": item.get("torque_coordinates", ""),
                "contacts": item.get("contacts", ""),
                "compare": item.get("compare", ""),
                "purpose": item.get("purpose", ""),
                "status": "planned",
            }
        )
    return rows


def _qvel_index(mjcf: Any, name: str) -> int | None:
    if name.startswith("qpos[") and name.endswith("]"):
        idx = int(name[5:-1])
        return idx if 0 <= idx < int(mjcf.model.nv) else None
    joint_id = mjcf.mujoco.mj_name2id(mjcf.model, mjcf.mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        return None
    return int(mjcf.model.jnt_dofadr[joint_id])


def _set_opensim_zero_state(osim: Any) -> None:
    _set_opensim_speeds(osim, {})
    _set_opensim_activations(osim, {})


def _set_opensim_speeds(osim: Any, speeds: dict[str, float]) -> None:
    coord_set = osim.model.getCoordinateSet()
    for i in range(coord_set.getSize()):
        try:
            coord = coord_set.get(i)
            coord.setSpeedValue(osim.state, float(speeds.get(coord.getName(), 0.0)))
        except Exception:
            continue


def _set_opensim_activations(osim: Any, activations: dict[str, float]) -> None:
    muscles = osim.model.getMuscles()
    for i in range(muscles.getSize()):
        muscle = muscles.get(i)
        try:
            muscle.setActivation(osim.state, float(activations.get(muscle.getName(), 0.0)))
        except Exception:
            pass
        try:
            muscle.computeEquilibrium(osim.state)
        except Exception:
            pass


def _mujoco_forward(
    mjcf: Any,
    pose: dict[str, float],
    speeds: dict[str, float] | None,
    activations: dict[str, float] | None,
    disable_contact: bool,
) -> None:
    mj = mjcf.mujoco
    model = mjcf.model
    data = mjcf.data
    old_disable = int(model.opt.disableflags)
    try:
        if disable_contact:
            model.opt.disableflags = old_disable | int(mj.mjtDisableBit.mjDSBL_CONTACT)
        data.qpos[:] = model.qpos0
        for name, value in pose.items():
            idx = mjcf.qpos_index(name)
            if idx is not None and 0 <= idx < model.nq:
                data.qpos[idx] = float(value)
        data.qvel[:] = 0.0
        for name, value in (speeds or {}).items():
            idx = _qvel_index(mjcf, name)
            if idx is not None and 0 <= idx < model.nv:
                data.qvel[idx] = float(value)
        if model.nu:
            data.ctrl[:] = 0.0
        if data.act is not None and data.act.size:
            data.act[:] = 0.0
        for name, value in (activations or {}).items():
            actuator_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id < 0:
                continue
            data.ctrl[actuator_id] = float(value)
            adr = int(model.actuator_actadr[actuator_id])
            num = int(model.actuator_actnum[actuator_id])
            if adr >= 0 and num > 0:
                data.act[adr : adr + num] = float(value)
        mj.mj_forward(model, data)
    finally:
        model.opt.disableflags = old_disable


def _mujoco_forward_zero_control(mjcf: Any, pose: dict[str, float], disable_contact: bool) -> None:
    _mujoco_forward(mjcf, pose, {}, {}, disable_contact)


def _opensim_locked(osim: Any, coordinate_name: str) -> bool:
    try:
        return bool(osim.model.getCoordinateSet().get(coordinate_name).getLocked(osim.state))
    except Exception:
        return False


def _independent_coordinate_items(osim: Any, mapping: MappingConfig) -> list[dict[str, Any]]:
    dependent = _dependent_coordinates(osim)
    items: list[dict[str, Any]] = []
    for item in mapping.coordinates:
        oname = mapping.side_name(item, "opensim")
        mname = mapping.side_name(item, "mujoco")
        if not oname or not mname or oname.startswith("root_") or mname == "root":
            continue
        if oname in dependent or _opensim_locked(osim, oname):
            continue
        items.append(item)
    return items


def _mapped_qacc_rows(osim: Any, mjcf: Any, mapping: MappingConfig, coordinate_items: list[dict[str, Any]], experiment: str) -> list[dict[str, Any]]:
    coord_set = osim.model.getCoordinateSet()
    rows: list[dict[str, Any]] = []
    for item in coordinate_items:
        oname = mapping.side_name(item, "opensim")
        mname = mapping.side_name(item, "mujoco")
        if not oname or not mname:
            continue
        try:
            qvel_idx = _qvel_index(mjcf, mname)
            if qvel_idx is None:
                raise KeyError(f"MuJoCo qvel index not found for {mname}")
            opensim_qacc = float(coord_set.get(oname).getAccelerationValue(osim.state))
            sign = -1.0 if bool(item.get("sign_flip", False)) else 1.0
            mujoco_qacc = sign * float(mjcf.data.qacc[qvel_idx])
            err = abs(opensim_qacc - mujoco_qacc)
            status = "evaluated"
        except Exception as exc:
            opensim_qacc = mujoco_qacc = err = np.nan
            status = f"skipped: {exc}"
        rows.append(
            {
                "experiment": experiment,
                "opensim_coordinate": oname,
                "mujoco_qvel": mname,
                "coordinate_role": "independent",
                "opensim_qacc": opensim_qacc,
                "mujoco_qacc": mujoco_qacc,
                "absolute_error": err,
                "relative_error": rel_error(err, max(abs(opensim_qacc), abs(mujoco_qacc))),
                "status": status,
            }
        )
    return rows


def _instant_acceleration_rows(osim: Any, mjcf: Any, mapping: MappingConfig) -> list[dict[str, Any]]:
    sample = next((s for s in mapping.pose_samples if str(s.get("name", "")) == "neutral"), mapping.pose_samples[0])
    constraints = _coupler_constraints(osim)
    opensim_pose = _with_coupled_coordinates(_pose_values(sample, mapping, "opensim"), constraints, mapping, "opensim")
    mujoco_pose = _with_coupled_coordinates(_pose_values(sample, mapping, "mujoco"), constraints, mapping, "mujoco")
    dependent = _dependent_coordinates(osim)

    osim.set_pose(opensim_pose)
    _set_opensim_zero_state(osim)
    try:
        osim.model.realizeAcceleration(osim.state)
    except Exception as exc:
        return [
            {
                "experiment": "passive_no_contact_instant_acceleration",
                "opensim_coordinate": "",
                "mujoco_qvel": "",
                "coordinate_role": "",
                "opensim_qacc": np.nan,
                "mujoco_qacc": np.nan,
                "absolute_error": np.nan,
                "relative_error": np.nan,
                "status": f"skipped: OpenSim acceleration realization failed: {exc}",
            }
        ]

    _mujoco_forward_zero_control(mjcf, mujoco_pose, disable_contact=True)
    rows: list[dict[str, Any]] = []
    comparable = {id(item): item for item in _independent_coordinate_items(osim, mapping)}
    for item in mapping.coordinates:
        oname = mapping.side_name(item, "opensim")
        mname = mapping.side_name(item, "mujoco")
        if not oname or not mname:
            continue
        role = "dependent" if oname in dependent else "independent"
        if role == "independent" and _opensim_locked(osim, oname):
            role = "locked"
        if id(item) not in comparable:
            rows.append(
                {
                    "experiment": "passive_no_contact_instant_acceleration",
                    "opensim_coordinate": oname,
                    "mujoco_qvel": mname,
                    "coordinate_role": role if role != "independent" else "root_or_freejoint",
                    "opensim_qacc": np.nan,
                    "mujoco_qacc": np.nan,
                    "absolute_error": np.nan,
                    "relative_error": np.nan,
                    "status": "skipped: not a directly comparable independent scalar coordinate",
                }
            )
            continue
        rows.extend(_mapped_qacc_rows(osim, mjcf, mapping, [item], "passive_no_contact_instant_acceleration"))
    return rows


def _activation_experiment(mapping: MappingConfig, name: str) -> dict[str, Any]:
    experiments = mapping.dynamics_experiments.get("muscle_torque", []) + mapping.dynamics_experiments.get("forward_dynamics", [])
    for item in experiments if isinstance(experiments, list) else []:
        if isinstance(item, dict) and item.get("name") == name:
            return item
    return {}


def _parse_range(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, str) and "-" in value:
        lo, hi = value.split("-", 1)
        return int(lo), int(hi)
    if isinstance(value, (int, float)):
        return int(value), int(value)
    return default


def _interp_curve(curve: list[tuple[float, float]], x: float) -> float:
    clean = sorted((float(a), float(b)) for a, b in curve if np.isfinite(a) and np.isfinite(b))
    if not clean:
        return 0.0
    if x <= clean[0][0]:
        return clean[0][1]
    if x >= clean[-1][0]:
        return clean[-1][1]
    for (x0, y0), (x1, y1) in zip(clean[:-1], clean[1:]):
        if x0 <= x <= x1:
            if abs(x1 - x0) < 1e-12:
                return y0
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return clean[-1][1]


def _build_activation_adapter(
    osim: Any,
    mjcf: Any,
    mapping: MappingConfig,
    opensim_pose: dict[str, float],
    mujoco_pose: dict[str, float],
    mujoco_actuator_filter: set[str] | None = None,
) -> dict[str, list[tuple[float, float]]]:
    adapter: dict[str, list[tuple[float, float]]] = {}
    for item in mapping.muscles:
        oname = mapping.side_name(item, "opensim")
        mname = mapping.side_name(item, "mujoco")
        if not oname or not mname:
            continue
        if mujoco_actuator_filter is not None and mname not in mujoco_actuator_filter:
            continue
        try:
            opensim_baseline = abs(_opensim_actuation(osim, oname, 0.0, opensim_pose))
            mujoco_baseline = _mujoco_actuator_force(mjcf, mname, 0.0, mujoco_pose)
            opensim_curve = [
                (activation, abs(_opensim_actuation(osim, oname, activation, opensim_pose)) - opensim_baseline)
                for activation in _calibration_values()
            ]
            adapter[oname] = [
                (
                    activation,
                    _interp_activation_for_force(
                        opensim_curve,
                        _mujoco_actuator_force(mjcf, mname, activation, mujoco_pose) - mujoco_baseline,
                    ),
                )
                for activation in _calibration_values()
            ]
        except Exception:
            adapter[oname] = [(0.0, 0.0), (1.0, 1.0)]
    return adapter


def _opensim_activation_values(mujoco_activations: dict[str, float], mapping: MappingConfig, adapter: dict[str, list[tuple[float, float]]]) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in mapping.muscles:
        oname = mapping.side_name(item, "opensim")
        mname = mapping.side_name(item, "mujoco")
        if not oname or not mname:
            continue
        values[oname] = _interp_curve(adapter.get(oname, [(0.0, 0.0), (1.0, 1.0)]), float(mujoco_activations.get(mname, 0.0)))
    return values


def _random_activation_vectors(mapping: MappingConfig) -> list[dict[str, float]]:
    experiment = _activation_experiment(mapping, "sparse_random_activation_vectors")
    adapter_cfg = mapping.raw.get("_adapter_config", {})
    adapter_random = adapter_cfg.get("random_activation", {}) if isinstance(adapter_cfg, dict) and isinstance(adapter_cfg.get("random_activation", {}), dict) else {}
    seed = int(adapter_random.get("seed", experiment.get("seed", 20260507)))
    vector_count = int(adapter_random.get("vector_count", experiment.get("vector_count", 20)))
    active_range = _parse_range(adapter_random.get("active_muscles_per_vector", experiment.get("active_muscles_per_vector")), (5, 10))
    names = [mapping.side_name(item, "mujoco") for item in mapping.muscles]
    names = [name for name in names if name]
    rng = np.random.default_rng(seed)
    vectors: list[dict[str, float]] = []
    if bool(adapter_random.get("include_single_muscle_basis", False)):
        basis_activation = float(adapter_random.get("basis_activation", 1.0))
        vectors.extend({str(name): basis_activation} for name in names)
    for _ in range(vector_count):
        count = int(rng.integers(active_range[0], active_range[1] + 1))
        count = max(0, min(count, len(names)))
        chosen = rng.choice(names, size=count, replace=False) if count else []
        vectors.append({str(name): float(rng.uniform(0.05, 1.0)) for name in chosen})
    return vectors


def _random_activation_qacc_rows(osim: Any, mjcf: Any, mapping: MappingConfig) -> list[dict[str, Any]]:
    sample = next((s for s in mapping.pose_samples if str(s.get("name", "")) == "neutral"), mapping.pose_samples[0])
    constraints = _coupler_constraints(osim)
    opensim_pose = _with_coupled_coordinates(_pose_values(sample, mapping, "opensim"), constraints, mapping, "opensim")
    mujoco_pose = _with_coupled_coordinates(_pose_values(sample, mapping, "mujoco"), constraints, mapping, "mujoco")
    coordinate_items = _independent_coordinate_items(osim, mapping)
    adapter = _build_activation_adapter(osim, mjcf, mapping, opensim_pose, mujoco_pose)
    rows: list[dict[str, Any]] = []
    vectors = _random_activation_vectors(mapping)
    debug = mapping.raw.get("_debug", {})
    debug_vector_index = debug.get("vector_index") if isinstance(debug, dict) else None
    if debug_vector_index is not None:
        idx = int(debug_vector_index)
        vectors = [vectors[idx]] if 0 <= idx < len(vectors) else []
        vector_indices = [idx] if vectors else []
    else:
        vector_indices = list(range(len(vectors)))
    for vector_idx, mujoco_activations in zip(vector_indices, vectors):
        opensim_activations = _opensim_activation_values(mujoco_activations, mapping, adapter)
        osim.set_pose(opensim_pose)
        _set_opensim_speeds(osim, {})
        _set_opensim_activations(osim, opensim_activations)
        try:
            osim.model.realizeAcceleration(osim.state)
        except Exception as exc:
            rows.append({"experiment": "sparse_random_activation_qacc", "vector_index": vector_idx, "status": f"skipped: {exc}"})
            continue
        _mujoco_forward(mjcf, mujoco_pose, {}, mujoco_activations, disable_contact=True)
        for row in _mapped_qacc_rows(osim, mjcf, mapping, coordinate_items, "sparse_random_activation_qacc"):
            row["vector_index"] = vector_idx
            row["active_muscle_count"] = len(mujoco_activations)
            rows.append(row)
    return rows


def _mujoco_mass_diag(mjcf: Any) -> np.ndarray:
    mj = mjcf.mujoco
    dense = np.zeros((int(mjcf.model.nv), int(mjcf.model.nv)), dtype=float)
    mj.mj_fullM(mjcf.model, dense, mjcf.data.qM)
    return np.diag(dense)


def _mujoco_dof_value(model: Any, field: str, qvel_idx: int) -> float:
    value = getattr(model, field, None)
    if value is None or qvel_idx < 0 or qvel_idx >= len(value):
        return np.nan
    return float(value[qvel_idx])


def _coordinate_force_probes(osim: Any) -> dict[str, Any]:
    """Return existing level4 coordinate-force probes keyed by coordinate."""
    probes: dict[str, Any] = {}
    try:
        cls = osim.opensim.ExpressionBasedCoordinateForce
        force_set = osim.model.getForceSet()
        for i in range(force_set.getSize()):
            force = cls.safeDownCast(force_set.get(i))
            if force and str(force.getName()).startswith("level4_neutral_qacc_fit_"):
                probes[str(force.getCoordinateName())] = force
    except Exception:
        return {}
    return probes


def _set_probe_expressions(probes: dict[str, Any], active_coordinate: str | None = None) -> None:
    for coordinate, force in probes.items():
        force.setExpression("1" if active_coordinate is not None and coordinate == active_coordinate else "0")


def _opensim_qacc_vector(osim: Any, coordinate_items: list[dict[str, Any]], mapping: MappingConfig) -> np.ndarray:
    coord_set = osim.model.getCoordinateSet()
    values: list[float] = []
    for item in coordinate_items:
        oname = mapping.side_name(item, "opensim")
        if not oname:
            values.append(np.nan)
            continue
        values.append(float(coord_set.get(oname).getAccelerationValue(osim.state)))
    return np.array(values, dtype=float)


def _mujoco_qacc_vector(mjcf: Any, coordinate_items: list[dict[str, Any]], mapping: MappingConfig) -> np.ndarray:
    values: list[float] = []
    for item in coordinate_items:
        mname = mapping.side_name(item, "mujoco")
        qvel_idx = _qvel_index(mjcf, mname or "")
        if qvel_idx is None:
            values.append(np.nan)
            continue
        sign = -1.0 if bool(item.get("sign_flip", False)) else 1.0
        values.append(sign * float(mjcf.data.qacc[qvel_idx]))
    return np.array(values, dtype=float)


def _mujoco_sensitivity_matrix(
    mjcf: Any,
    pose: dict[str, float],
    coordinate_items: list[dict[str, Any]],
    mapping: MappingConfig,
    *,
    armature_scale: float,
) -> np.ndarray:
    mj = mjcf.mujoco
    model = mjcf.model
    data = mjcf.data
    original_armature = np.array(model.dof_armature, dtype=float).copy() if hasattr(model, "dof_armature") else None
    old_disable = int(model.opt.disableflags)
    qvel_indices = [_qvel_index(mjcf, mapping.side_name(item, "mujoco") or "") for item in coordinate_items]
    signs = np.array([-1.0 if bool(item.get("sign_flip", False)) else 1.0 for item in coordinate_items], dtype=float)

    def set_state() -> None:
        data.qpos[:] = model.qpos0
        for name, value in pose.items():
            idx = mjcf.qpos_index(name)
            if idx is not None and 0 <= idx < model.nq:
                data.qpos[idx] = float(value)
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        data.qfrc_applied[:] = 0.0
        if model.nu:
            data.ctrl[:] = 0.0
        if data.act is not None and data.act.size:
            data.act[:] = 0.0
        model.opt.disableflags = old_disable | int(mj.mjtDisableBit.mjDSBL_CONTACT)
        mj.mj_forward(model, data)

    try:
        if original_armature is not None:
            model.dof_armature[:] = original_armature * float(armature_scale)
        set_state()
        baseline = _mujoco_qacc_vector(mjcf, coordinate_items, mapping)
        sensitivity = np.full((len(coordinate_items), len(coordinate_items)), np.nan, dtype=float)
        for col, qvel_idx in enumerate(qvel_indices):
            if qvel_idx is None:
                continue
            set_state()
            data.qfrc_applied[qvel_idx] = signs[col]
            mj.mj_forward(model, data)
            sensitivity[:, col] = _mujoco_qacc_vector(mjcf, coordinate_items, mapping) - baseline
        return sensitivity
    finally:
        model.opt.disableflags = old_disable
        data.qfrc_applied[:] = 0.0
        if original_armature is not None:
            model.dof_armature[:] = original_armature
        mj.mj_forward(model, data)


def _level4_probe_context(osim: Any, mjcf: Any, mapping: MappingConfig) -> tuple[Any, dict[str, Any], dict[str, float], dict[str, float], list[dict[str, Any]]] | None:
    if not getattr(osim, "path", None):
        return None
    probe_osim = OpenSimModel.load(osim.path)
    probes = _coordinate_force_probes(probe_osim)
    if not probes:
        return None
    sample = next((s for s in mapping.pose_samples if str(s.get("name", "")) == "neutral"), mapping.pose_samples[0])
    constraints = _coupler_constraints(probe_osim)
    opensim_pose = _with_coupled_coordinates(_pose_values(sample, mapping, "opensim"), constraints, mapping, "opensim")
    mujoco_pose = _with_coupled_coordinates(_pose_values(sample, mapping, "mujoco"), constraints, mapping, "mujoco")
    coordinate_items = _independent_coordinate_items(probe_osim, mapping)
    missing = [mapping.side_name(item, "opensim") for item in coordinate_items if mapping.side_name(item, "opensim") not in probes]
    if missing:
        return None
    return probe_osim, probes, opensim_pose, mujoco_pose, coordinate_items


def _opensim_sensitivity_matrix(
    probe_osim: Any,
    probes: dict[str, Any],
    opensim_pose: dict[str, float],
    coordinate_items: list[dict[str, Any]],
    mapping: MappingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    _set_probe_expressions(probes)
    probe_osim.state = probe_osim.model.initSystem()
    probe_osim.set_pose(opensim_pose)
    _set_opensim_speeds(probe_osim, {})
    _set_opensim_activations(probe_osim, {})
    probe_osim.model.realizeAcceleration(probe_osim.state)
    baseline = _opensim_qacc_vector(probe_osim, coordinate_items, mapping)

    sensitivity = np.full((len(coordinate_items), len(coordinate_items)), np.nan, dtype=float)
    for col, item in enumerate(coordinate_items):
        oname = mapping.side_name(item, "opensim")
        if not oname:
            continue
        _set_probe_expressions(probes, oname)
        probe_osim.state = probe_osim.model.initSystem()
        probe_osim.set_pose(opensim_pose)
        _set_opensim_speeds(probe_osim, {})
        _set_opensim_activations(probe_osim, {})
        probe_osim.model.realizeAcceleration(probe_osim.state)
        sensitivity[:, col] = _opensim_qacc_vector(probe_osim, coordinate_items, mapping) - baseline
    _set_probe_expressions(probes)
    probe_osim.state = probe_osim.model.initSystem()
    return sensitivity, baseline


def _sensitivity_audit_rows(
    osim: Any,
    mjcf: Any,
    mapping: MappingConfig,
) -> tuple[
    list[dict[str, Any]],
    dict[str, float],
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    tuple[Any, dict[str, Any], dict[str, float], dict[str, float], list[dict[str, Any]]] | None,
]:
    context = _level4_probe_context(osim, mjcf, mapping)
    if context is None:
        return [], {}, None, None, None, None
    probe_osim, probes, opensim_pose, mujoco_pose, coordinate_items = context
    try:
        opensim_sensitivity, _ = _opensim_sensitivity_matrix(probe_osim, probes, opensim_pose, coordinate_items, mapping)
        mujoco_no_armature = _mujoco_sensitivity_matrix(mjcf, mujoco_pose, coordinate_items, mapping, armature_scale=0.0)
        mujoco_with_armature = _mujoco_sensitivity_matrix(mjcf, mujoco_pose, coordinate_items, mapping, armature_scale=1.0)
    except Exception:
        return [], {}, None, None, None, context

    qvel_indices = [_qvel_index(mjcf, mapping.side_name(item, "mujoco") or "") for item in coordinate_items]
    armature = np.array(
        [float(mjcf.model.dof_armature[idx]) if idx is not None and hasattr(mjcf.model, "dof_armature") else 0.0 for idx in qvel_indices],
        dtype=float,
    )
    lhs = np.eye(len(coordinate_items)) + opensim_sensitivity @ np.diag(armature)
    try:
        armature_transform = np.linalg.solve(lhs, np.eye(len(coordinate_items)))
    except np.linalg.LinAlgError:
        armature_transform = np.linalg.pinv(lhs, rcond=1e-12)
    armature_adjusted = armature_transform @ opensim_sensitivity

    def fro_rel(lhs: np.ndarray, rhs: np.ndarray) -> float:
        return float(np.linalg.norm(lhs - rhs) / max(np.linalg.norm(rhs), 1e-12))

    metrics = {
        "sensitivity_no_armature_fro_relative_error": fro_rel(opensim_sensitivity, mujoco_no_armature),
        "sensitivity_armature_adjusted_fro_relative_error": fro_rel(armature_adjusted, mujoco_with_armature),
    }
    rows: list[dict[str, Any]] = []
    for row, item in enumerate(coordinate_items):
        oname = mapping.side_name(item, "opensim")
        mname = mapping.side_name(item, "mujoco")
        os_diag = float(opensim_sensitivity[row, row])
        mj_no_diag = float(mujoco_no_armature[row, row])
        adjusted_diag = float(armature_adjusted[row, row])
        mj_with_diag = float(mujoco_with_armature[row, row])
        rows.append(
            {
                "opensim_coordinate": oname,
                "mujoco_qvel": mname,
                "mujoco_dof_armature": float(armature[row]),
                "opensim_no_armature_sensitivity_diag": os_diag,
                "mujoco_no_armature_sensitivity_diag": mj_no_diag,
                "no_armature_diag_abs_error": abs(os_diag - mj_no_diag),
                "no_armature_diag_relative_error": rel_error(abs(os_diag - mj_no_diag), max(abs(os_diag), abs(mj_no_diag))),
                "opensim_armature_adjusted_sensitivity_diag": adjusted_diag,
                "mujoco_with_armature_sensitivity_diag": mj_with_diag,
                "armature_adjusted_diag_abs_error": abs(adjusted_diag - mj_with_diag),
                "armature_adjusted_diag_relative_error": rel_error(abs(adjusted_diag - mj_with_diag), max(abs(adjusted_diag), abs(mj_with_diag))),
                "status": "evaluated",
            }
        )
    rows.sort(key=lambda row: float(row["armature_adjusted_diag_abs_error"]), reverse=True)
    return rows, metrics, armature_transform, opensim_sensitivity, mujoco_with_armature, context


def _armature_adjusted_random_qacc_rows(
    mjcf: Any,
    mapping: MappingConfig,
    armature_transform: np.ndarray | None,
    opensim_sensitivity: np.ndarray | None,
    context: tuple[Any, dict[str, Any], dict[str, float], dict[str, float], list[dict[str, Any]]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if armature_transform is None or context is None:
        return [], []
    probe_osim, probes, opensim_pose, mujoco_pose, coordinate_items = context
    _set_probe_expressions(probes)
    probe_osim.state = probe_osim.model.initSystem()
    adapter = _build_activation_adapter(probe_osim, mjcf, mapping, opensim_pose, mujoco_pose)
    rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    coord_names = [mapping.side_name(item, "opensim") for item in coordinate_items]
    qvel_indices = [_qvel_index(mjcf, mapping.side_name(item, "mujoco") or "") for item in coordinate_items]
    armature = np.array(
        [float(mjcf.model.dof_armature[idx]) if idx is not None and hasattr(mjcf.model, "dof_armature") else 0.0 for idx in qvel_indices],
        dtype=float,
    )
    sensitivity_inverse = np.linalg.pinv(opensim_sensitivity, rcond=1e-12) if opensim_sensitivity is not None else None
    for vector_idx, mujoco_activations in enumerate(_random_activation_vectors(mapping)):
        opensim_activations = _opensim_activation_values(mujoco_activations, mapping, adapter)
        probe_osim.set_pose(opensim_pose)
        _set_opensim_speeds(probe_osim, {})
        _set_opensim_activations(probe_osim, opensim_activations)
        try:
            probe_osim.model.realizeAcceleration(probe_osim.state)
            opensim_raw = _opensim_qacc_vector(probe_osim, coordinate_items, mapping)
            opensim_adjusted = armature_transform @ opensim_raw
        except Exception as exc:
            rows.append({"experiment": "diagnostic_armature_adjusted_sparse_random_activation_qacc", "vector_index": vector_idx, "status": f"skipped: {exc}"})
            continue
        _mujoco_forward(mjcf, mujoco_pose, {}, mujoco_activations, disable_contact=True)
        mujoco_qacc = _mujoco_qacc_vector(mjcf, coordinate_items, mapping)
        if sensitivity_inverse is not None:
            opensim_force_estimate = sensitivity_inverse @ opensim_raw
            mujoco_force_estimate = sensitivity_inverse @ mujoco_qacc + armature * mujoco_qacc
            force_residual = opensim_force_estimate - mujoco_force_estimate
        else:
            opensim_force_estimate = mujoco_force_estimate = force_residual = np.full(len(coordinate_items), np.nan, dtype=float)
        for row_idx, item in enumerate(coordinate_items):
            oname = coord_names[row_idx]
            mname = mapping.side_name(item, "mujoco")
            err = abs(float(opensim_adjusted[row_idx]) - float(mujoco_qacc[row_idx]))
            rows.append(
                {
                    "experiment": "diagnostic_armature_adjusted_sparse_random_activation_qacc",
                    "opensim_coordinate": oname,
                    "mujoco_qvel": mname,
                    "coordinate_role": "independent",
                    "opensim_qacc": float(opensim_adjusted[row_idx]),
                    "opensim_raw_no_correction_qacc": float(opensim_raw[row_idx]),
                    "mujoco_qacc": float(mujoco_qacc[row_idx]),
                    "absolute_error": err,
                    "relative_error": rel_error(err, max(abs(float(opensim_adjusted[row_idx])), abs(float(mujoco_qacc[row_idx])))),
                    "status": "evaluated",
                    "vector_index": vector_idx,
                    "active_muscle_count": len(mujoco_activations),
                    "diagnostic_only": True,
                }
            )
            residual_rows.append(
                {
                    "experiment": "diagnostic_sparse_random_activation_generalized_force_residual",
                    "opensim_coordinate": oname,
                    "mujoco_qvel": mname,
                    "opensim_generalized_force_estimate": float(opensim_force_estimate[row_idx]),
                    "mujoco_generalized_force_estimate": float(mujoco_force_estimate[row_idx]),
                    "generalized_force_residual": float(force_residual[row_idx]),
                    "absolute_force_residual": abs(float(force_residual[row_idx])),
                    "opensim_raw_no_correction_qacc": float(opensim_raw[row_idx]),
                    "mujoco_qacc": float(mujoco_qacc[row_idx]),
                    "vector_index": vector_idx,
                    "active_muscle_count": len(mujoco_activations),
                    "status": "evaluated" if np.isfinite(force_residual[row_idx]) else "skipped: sensitivity inverse unavailable",
                    "diagnostic_only": True,
                    "note": "Estimated with the OpenSim no-armature acceleration sensitivity inverse; use for source ranking, not as a pass/fail gate.",
                }
            )
    return rows, residual_rows


def _mass_adapter_random_qacc_rows(
    mjcf: Any,
    mapping: MappingConfig,
    opensim_sensitivity: np.ndarray | None,
    mujoco_with_armature_sensitivity: np.ndarray | None,
    context: tuple[Any, dict[str, Any], dict[str, float], dict[str, float], list[dict[str, Any]]] | None,
) -> list[dict[str, Any]]:
    if opensim_sensitivity is None or mujoco_with_armature_sensitivity is None or context is None:
        return []
    probe_osim, probes, opensim_pose, mujoco_pose, coordinate_items = context
    _set_probe_expressions(probes)
    probe_osim.state = probe_osim.model.initSystem()
    try:
        dynamics_adapter = mujoco_with_armature_sensitivity @ np.linalg.pinv(opensim_sensitivity, rcond=1e-12)
    except Exception:
        return []

    rows: list[dict[str, Any]] = []
    vectors = _random_activation_vectors(mapping)
    debug = mapping.raw.get("_debug", {})
    debug_vector_index = debug.get("vector_index") if isinstance(debug, dict) else None
    if debug_vector_index is not None:
        idx = int(debug_vector_index)
        vectors = [vectors[idx]] if 0 <= idx < len(vectors) else []
        vector_indices = [idx] if vectors else []
    else:
        vector_indices = list(range(len(vectors)))
    active_actuators = {name for vector in vectors for name in vector}
    adapter = _build_activation_adapter(probe_osim, mjcf, mapping, opensim_pose, mujoco_pose, active_actuators)
    for vector_idx, mujoco_activations in zip(vector_indices, vectors):
        opensim_activations = _opensim_activation_values(mujoco_activations, mapping, adapter)
        probe_osim.set_pose(opensim_pose)
        _set_opensim_speeds(probe_osim, {})
        _set_opensim_activations(probe_osim, opensim_activations)
        try:
            probe_osim.model.realizeAcceleration(probe_osim.state)
            opensim_raw = _opensim_qacc_vector(probe_osim, coordinate_items, mapping)
            opensim_adapter_qacc = dynamics_adapter @ opensim_raw
        except Exception as exc:
            rows.append({"experiment": "diagnostic_mass_adapter_sparse_random_activation_qacc", "vector_index": vector_idx, "status": f"skipped: {exc}"})
            continue
        _mujoco_forward(mjcf, mujoco_pose, {}, mujoco_activations, disable_contact=True)
        mujoco_qacc = _mujoco_qacc_vector(mjcf, coordinate_items, mapping)
        for row_idx, item in enumerate(coordinate_items):
            oname = mapping.side_name(item, "opensim")
            mname = mapping.side_name(item, "mujoco")
            err = abs(float(opensim_adapter_qacc[row_idx]) - float(mujoco_qacc[row_idx]))
            rows.append(
                {
                    "experiment": "diagnostic_mass_adapter_sparse_random_activation_qacc",
                    "opensim_coordinate": oname,
                    "mujoco_qvel": mname,
                    "coordinate_role": "independent",
                    "opensim_mass_adapter_qacc": float(opensim_adapter_qacc[row_idx]),
                    "opensim_raw_no_correction_qacc": float(opensim_raw[row_idx]),
                    "mujoco_qacc": float(mujoco_qacc[row_idx]),
                    "absolute_error": err,
                    "relative_error": rel_error(err, max(abs(float(opensim_adapter_qacc[row_idx])), abs(float(mujoco_qacc[row_idx])))),
                    "status": "evaluated",
                    "vector_index": vector_idx,
                    "active_muscle_count": len(mujoco_activations),
                    "diagnostic_only": True,
                    "note": "Diagnostic only: maps OpenSim raw qacc through MuJoCo with-armature generalized-force sensitivity. This is not a native OpenSim model gate.",
                }
            )
    return rows


def _qacc_error_lookup(random_df: Any) -> dict[str, dict[str, float]]:
    lookup: dict[str, dict[str, float]] = {}
    if random_df.empty:
        return lookup
    finite = random_df[random_df["status"] == "evaluated"].copy()
    if finite.empty:
        return lookup
    finite["absolute_error"] = finite["absolute_error"].astype(float)
    for coordinate, group in finite.groupby("opensim_coordinate"):
        worst = group.sort_values("absolute_error", ascending=False).iloc[0]
        lookup[str(coordinate)] = {
            "worst_random_activation_qacc_error": float(worst["absolute_error"]),
            "worst_vector_index": float(worst.get("vector_index", np.nan)),
            "worst_opensim_qacc": float(worst["opensim_qacc"]),
            "worst_mujoco_qacc": float(worst["mujoco_qacc"]),
        }
    return lookup


def _coordinate_dynamics_audit_rows(mjcf: Any, mapping: MappingConfig, coordinate_items: list[dict[str, Any]], random_df: Any) -> list[dict[str, Any]]:
    mass_diag = _mujoco_mass_diag(mjcf)
    qacc_lookup = _qacc_error_lookup(random_df)
    rows: list[dict[str, Any]] = []
    for item in coordinate_items:
        oname = mapping.side_name(item, "opensim")
        mname = mapping.side_name(item, "mujoco")
        if not oname or not mname:
            continue
        qvel_idx = _qvel_index(mjcf, mname)
        if qvel_idx is None:
            rows.append(
                {
                    "opensim_coordinate": oname,
                    "mujoco_qvel": mname,
                    "qvel_idx": "",
                    "status": "skipped: MuJoCo qvel index not found",
                }
            )
            continue
        armature = _mujoco_dof_value(mjcf.model, "dof_armature", qvel_idx)
        damping = _mujoco_dof_value(mjcf.model, "dof_damping", qvel_idx)
        frictionloss = _mujoco_dof_value(mjcf.model, "dof_frictionloss", qvel_idx)
        total_mass_diag = float(mass_diag[qvel_idx]) if qvel_idx < len(mass_diag) else np.nan
        physical_mass_diag_without_armature = total_mass_diag - armature if np.isfinite(total_mass_diag) and np.isfinite(armature) else np.nan
        lookup = qacc_lookup.get(oname, {})
        rows.append(
            {
                "opensim_coordinate": oname,
                "mujoco_qvel": mname,
                "qvel_idx": qvel_idx,
                "mujoco_dof_armature": armature,
                "mujoco_dof_damping": damping,
                "mujoco_dof_frictionloss": frictionloss,
                "mujoco_mass_diag_total": total_mass_diag,
                "mujoco_mass_diag_minus_armature": physical_mass_diag_without_armature,
                "armature_fraction_of_mass_diag": armature / total_mass_diag if np.isfinite(total_mass_diag) and abs(total_mass_diag) > 1e-12 else np.nan,
                "worst_random_activation_qacc_error": lookup.get("worst_random_activation_qacc_error", np.nan),
                "worst_vector_index": lookup.get("worst_vector_index", np.nan),
                "worst_opensim_qacc": lookup.get("worst_opensim_qacc", np.nan),
                "worst_mujoco_qacc": lookup.get("worst_mujoco_qacc", np.nan),
                "status": "evaluated",
            }
        )
    rows.sort(key=lambda row: float(row.get("worst_random_activation_qacc_error", -np.inf)) if np.isfinite(row.get("worst_random_activation_qacc_error", np.nan)) else -np.inf, reverse=True)
    return rows


def _random_activation_qacc_without_mujoco_armature_rows(osim: Any, mjcf: Any, mapping: MappingConfig) -> list[dict[str, Any]]:
    if not hasattr(mjcf.model, "dof_armature"):
        return []
    original = np.array(mjcf.model.dof_armature, dtype=float).copy()
    try:
        mjcf.model.dof_armature[:] = 0.0
        rows = _random_activation_qacc_rows(osim, mjcf, mapping)
        for row in rows:
            row["experiment"] = "diagnostic_sparse_random_activation_qacc_mujoco_armature_zeroed"
            row["diagnostic_only"] = True
        return rows
    finally:
        mjcf.model.dof_armature[:] = original


def _state_error_rows(
    step: int,
    time_s: float,
    q_osim: dict[str, float],
    qdot_osim: dict[str, float],
    q_mujoco: dict[str, float],
    qdot_mujoco: dict[str, float],
    coordinate_items: list[dict[str, Any]],
    mapping: MappingConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in coordinate_items:
        oname = mapping.side_name(item, "opensim")
        mname = mapping.side_name(item, "mujoco")
        if not oname or not mname:
            continue
        sign = -1.0 if bool(item.get("sign_flip", False)) else 1.0
        q_err = abs(float(q_osim.get(oname, 0.0)) - sign * float(q_mujoco.get(mname, 0.0)))
        qdot_err = abs(float(qdot_osim.get(oname, 0.0)) - sign * float(qdot_mujoco.get(mname, 0.0)))
        rows.append(
            {
                "experiment": "matched_activation_rollout_200ms",
                "step": step,
                "time_s": time_s,
                "opensim_coordinate": oname,
                "mujoco_qvel": mname,
                "q_error": q_err,
                "qdot_error": qdot_err,
                "status": "evaluated",
            }
        )
    return rows


def _rollout_rows(osim: Any, mjcf: Any, mapping: MappingConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    experiment = _activation_experiment(mapping, "matched_activation_rollout_200ms")
    horizon = float(experiment.get("horizon_s", 0.2))
    dt = float(experiment.get("dt_s", 0.001))
    steps = max(1, int(round(horizon / dt)))
    sample = next((s for s in mapping.pose_samples if str(s.get("name", "")) == "neutral"), mapping.pose_samples[0])
    constraints = _coupler_constraints(osim)
    opensim_pose = _with_coupled_coordinates(_pose_values(sample, mapping, "opensim"), constraints, mapping, "opensim")
    mujoco_pose = _with_coupled_coordinates(_pose_values(sample, mapping, "mujoco"), constraints, mapping, "mujoco")
    coordinate_items = _independent_coordinate_items(osim, mapping)
    adapter = _build_activation_adapter(osim, mjcf, mapping, opensim_pose, mujoco_pose)
    random_vectors = _random_activation_vectors(mapping)
    mujoco_activations = random_vectors[0] if random_vectors else {}
    opensim_activations = _opensim_activation_values(mujoco_activations, mapping, adapter)

    q_osim = {mapping.side_name(item, "opensim"): opensim_pose.get(mapping.side_name(item, "opensim"), 0.0) for item in coordinate_items}
    q_mujoco = {mapping.side_name(item, "mujoco"): mujoco_pose.get(mapping.side_name(item, "mujoco"), 0.0) for item in coordinate_items}
    qdot_osim = {name: 0.0 for name in q_osim if name}
    qdot_mujoco = {name: 0.0 for name in q_mujoco if name}
    state_rows: list[dict[str, Any]] = []
    qacc_rows: list[dict[str, Any]] = []
    for step in range(steps + 1):
        time_s = step * dt
        opensim_full_pose = dict(opensim_pose)
        mujoco_full_pose = dict(mujoco_pose)
        opensim_full_pose.update({k: v for k, v in q_osim.items() if k})
        mujoco_full_pose.update({k: v for k, v in q_mujoco.items() if k})
        opensim_full_pose = _with_coupled_coordinates(opensim_full_pose, constraints, mapping, "opensim")
        mujoco_full_pose = _with_coupled_coordinates(mujoco_full_pose, constraints, mapping, "mujoco")

        osim.set_pose(opensim_full_pose)
        _set_opensim_speeds(osim, qdot_osim)
        _set_opensim_activations(osim, opensim_activations)
        try:
            osim.model.realizeAcceleration(osim.state)
        except Exception as exc:
            qacc_rows.append({"experiment": "matched_activation_rollout_200ms_qacc", "step": step, "status": f"skipped: {exc}"})
            break
        _mujoco_forward(mjcf, mujoco_full_pose, qdot_mujoco, mujoco_activations, disable_contact=True)
        current_qacc_rows = _mapped_qacc_rows(osim, mjcf, mapping, coordinate_items, "matched_activation_rollout_200ms_qacc")
        for row in current_qacc_rows:
            row["step"] = step
            row["time_s"] = time_s
        qacc_rows.extend(current_qacc_rows)
        state_rows.extend(_state_error_rows(step, time_s, q_osim, qdot_osim, q_mujoco, qdot_mujoco, coordinate_items, mapping))
        if step == steps:
            break
        for row in current_qacc_rows:
            if row.get("status") != "evaluated":
                continue
            oname = str(row["opensim_coordinate"])
            mname = str(row["mujoco_qvel"])
            sign = -1.0 if any(mapping.side_name(item, "opensim") == oname and bool(item.get("sign_flip", False)) for item in coordinate_items) else 1.0
            qdot_osim[oname] = float(qdot_osim.get(oname, 0.0)) + float(row["opensim_qacc"]) * dt
            qdot_mujoco[mname] = float(qdot_mujoco.get(mname, 0.0)) + sign * float(row["mujoco_qacc"]) * dt
            q_osim[oname] = float(q_osim.get(oname, 0.0)) + qdot_osim[oname] * dt
            q_mujoco[mname] = float(q_mujoco.get(mname, 0.0)) + qdot_mujoco[mname] * dt
    return state_rows, qacc_rows


def _status(df: Any, mapping: MappingConfig) -> str:
    if df.empty:
        return "not evaluated"
    finite = df[(df["status"] == "evaluated") & (df["coordinate_role"] == "independent")]
    if finite.empty:
        return "not evaluated"
    abs_err = np.array(finite["absolute_error"], dtype=float)
    rel_err = np.array(finite["relative_error"], dtype=float)
    reference = np.maximum(np.abs(np.array(finite["opensim_qacc"], dtype=float)), np.abs(np.array(finite["mujoco_qacc"], dtype=float)))
    warn_abs = float(mapping.thresholds.get("forward_dynamics_accel_warning", 5.0))
    fail_abs = float(mapping.thresholds.get("forward_dynamics_accel_fail", 20.0))
    warn_rel = float(mapping.thresholds.get("forward_dynamics_accel_warning_rel", 0.05))
    fail_rel = float(mapping.thresholds.get("forward_dynamics_accel_fail_rel", 0.10))
    if np.nanmax(abs_err) > fail_abs or np.nanmax(rel_err[np.isfinite(rel_err)]) > fail_rel:
        return "failed"
    if np.nanmax(abs_err) > warn_abs or np.nanmax(rel_err[np.isfinite(rel_err)]) > warn_rel:
        return "warning"
    return "passed"


def _rollout_status(df: Any, mapping: MappingConfig) -> str:
    if df.empty:
        return "not evaluated"
    finite = df[df["status"] == "evaluated"].copy()
    if finite.empty:
        return "not evaluated"
    q_err = np.array(finite["q_error"], dtype=float)
    qdot_err = np.array(finite["qdot_error"], dtype=float)
    q_warn = float(mapping.thresholds.get("forward_rollout_q_warning", 0.01))
    q_fail = float(mapping.thresholds.get("forward_rollout_q_fail", 0.05))
    qdot_warn = float(mapping.thresholds.get("forward_rollout_qdot_warning", 0.5))
    qdot_fail = float(mapping.thresholds.get("forward_rollout_qdot_fail", 2.0))
    if np.nanmax(q_err) > q_fail or np.nanmax(qdot_err) > qdot_fail:
        return "failed"
    if np.nanmax(q_err) > q_warn or np.nanmax(qdot_err) > qdot_warn:
        return "warning"
    return "passed"


def _combined_status(statuses: list[str]) -> str:
    if any(status == "failed" for status in statuses):
        return "failed"
    if any(status == "warning" for status in statuses):
        return "warning"
    if all(status == "not evaluated" for status in statuses):
        return "not evaluated"
    return "passed"


def _finite_qacc(df: Any) -> Any:
    return df[(df["status"] == "evaluated") & (df["coordinate_role"] == "independent")] if not df.empty else df


def run(osim: Any, mjcf: Any, mapping: MappingConfig, out_dir: Path) -> dict[str, Any]:
    planned = _plan_rows(mapping)
    if planned:
        write_csv(out_dir / "forward_dynamics_plan.csv", planned)
    rows = _instant_acceleration_rows(osim, mjcf, mapping)
    df = write_csv(out_dir / "forward_dynamics_smoke_test.csv", rows)
    random_rows = _random_activation_qacc_rows(osim, mjcf, mapping)
    random_df = write_csv(out_dir / "forward_dynamics_random_activation_qacc.csv", random_rows)
    coordinate_items = _independent_coordinate_items(osim, mapping)
    audit_rows = _coordinate_dynamics_audit_rows(mjcf, mapping, coordinate_items, random_df)
    audit_df = write_csv(out_dir / "diagnostics" / "mujoco_coordinate_dynamics_audit.csv", audit_rows)
    no_armature_rows = _random_activation_qacc_without_mujoco_armature_rows(osim, mjcf, mapping)
    no_armature_df = write_csv(out_dir / "diagnostics" / "forward_dynamics_random_activation_qacc_mujoco_armature_zeroed.csv", no_armature_rows)
    sensitivity_rows, sensitivity_metrics, armature_transform, opensim_sensitivity, mujoco_with_armature_sensitivity, probe_context = _sensitivity_audit_rows(osim, mjcf, mapping)
    sensitivity_df = write_csv(out_dir / "diagnostics" / "forward_dynamics_sensitivity_audit.csv", sensitivity_rows)
    armature_adjusted_rows, residual_rows = _armature_adjusted_random_qacc_rows(mjcf, mapping, armature_transform, opensim_sensitivity, probe_context)
    armature_adjusted_df = write_csv(out_dir / "diagnostics" / "forward_dynamics_armature_adjusted_random_activation_qacc.csv", armature_adjusted_rows)
    mass_adapter_rows = _mass_adapter_random_qacc_rows(mjcf, mapping, opensim_sensitivity, mujoco_with_armature_sensitivity, probe_context)
    mass_adapter_df = write_csv(out_dir / "diagnostics" / "forward_dynamics_mass_adapter_random_activation_qacc.csv", mass_adapter_rows)
    residual_df = write_csv(out_dir / "diagnostics" / "forward_dynamics_generalized_force_residual.csv", residual_rows)
    rollout_state_rows, rollout_qacc_rows = _rollout_rows(osim, mjcf, mapping)
    rollout_state_df = write_csv(out_dir / "forward_dynamics_rollout_state_error.csv", rollout_state_rows)
    rollout_qacc_df = write_csv(out_dir / "forward_dynamics_rollout_qacc_error.csv", rollout_qacc_rows)
    write_markdown(
        out_dir / "forward_dynamics_notes.md",
        "# Forward Dynamics Notes\n\n"
        "The executable gate compares t=0 generalized accelerations at the matched neutral pose with zero speeds and zero controls.\n"
        "MuJoCo contact is disabled for this no-contact smoke test. This avoids contact discontinuities and catches force, sign, inertia, "
        "coordinate-adapter and passive-actuation mismatches before long rollouts amplify them.\n"
        "The stricter random-activation gate samples sparse muscle activation vectors using the configured seed and compares generalized "
        "accelerations across all directly comparable independent coordinates.\n"
        "The matched-activation rollout uses a fixed sparse random activation vector, disables contact, recomputes accelerations each step, "
        "and integrates independent coordinates with the same semi-implicit Euler scheme on both sides. This is a deterministic short-horizon "
        "drift test, not a proof that arbitrary long RL rollouts remain bitwise identical.\n\n"
        "The diagnostics include MuJoCo per-DOF armature, damping, friction loss, and mass-matrix diagonal values. A separate diagnostic reruns "
        "the random-activation qacc comparison with MuJoCo armature zeroed in memory. That diagnostic does not modify the MuJoCo XML and is not "
        "used as a pass/fail gate; it only identifies whether the strict gate is dominated by MuJoCo simulator-level armature regularization.\n"
        "The sensitivity audit uses the existing level4_neutral_qacc_fit_* coordinate-force probes in an isolated reloaded OpenSim model with "
        "those correction forces zeroed. It estimates the independent-coordinate acceleration sensitivity matrix, compares it with MuJoCo unit "
        "generalized-force sensitivity with and without armature, and reports an armature-adjusted OpenSim qacc diagnostic. These diagnostics are "
        "not a gate and do not hide the raw strict full-body result. The generalized-force residual diagnostic uses the OpenSim no-armature "
        "sensitivity inverse to rank likely residual force/bias sources; it is an approximate source-ranking tool, not a pass/fail gate.\n"
        "The mass-adapter diagnostic maps OpenSim raw qacc through an explicit MuJoCo with-armature generalized-force sensitivity adapter. "
        "It is diagnostic-only and represents an external runtime dynamics adapter, not native OpenSim `.osim` behavior.\n",
    )
    files = [
        "forward_dynamics_smoke_test.csv",
        "forward_dynamics_random_activation_qacc.csv",
        "forward_dynamics_rollout_state_error.csv",
        "forward_dynamics_rollout_qacc_error.csv",
        "forward_dynamics_notes.md",
        "diagnostics/mujoco_coordinate_dynamics_audit.csv",
        "diagnostics/forward_dynamics_random_activation_qacc_mujoco_armature_zeroed.csv",
        "diagnostics/forward_dynamics_sensitivity_audit.csv",
        "diagnostics/forward_dynamics_armature_adjusted_random_activation_qacc.csv",
        "diagnostics/forward_dynamics_mass_adapter_random_activation_qacc.csv",
        "diagnostics/forward_dynamics_generalized_force_residual.csv",
    ]
    worst = write_worst_csv(out_dir / "diagnostics" / "forward_dynamics_worst_acceleration_error.csv", df, "absolute_error")
    if worst:
        files.append(f"diagnostics/{worst}")
    random_worst = write_worst_csv(out_dir / "diagnostics" / "forward_dynamics_random_activation_worst_qacc_error.csv", random_df, "absolute_error")
    if random_worst:
        files.append(f"diagnostics/{random_worst}")
    audit_worst = write_worst_csv(out_dir / "diagnostics" / "mujoco_coordinate_dynamics_audit_worst_qacc_error.csv", audit_df, "worst_random_activation_qacc_error")
    if audit_worst:
        files.append(f"diagnostics/{audit_worst}")
    no_armature_worst = write_worst_csv(out_dir / "diagnostics" / "forward_dynamics_random_activation_mujoco_armature_zeroed_worst_qacc_error.csv", no_armature_df, "absolute_error")
    if no_armature_worst:
        files.append(f"diagnostics/{no_armature_worst}")
    sensitivity_worst = write_worst_csv(out_dir / "diagnostics" / "forward_dynamics_sensitivity_worst_error.csv", sensitivity_df, "armature_adjusted_diag_abs_error")
    if sensitivity_worst:
        files.append(f"diagnostics/{sensitivity_worst}")
    armature_adjusted_worst = write_worst_csv(out_dir / "diagnostics" / "forward_dynamics_armature_adjusted_random_activation_worst_qacc_error.csv", armature_adjusted_df, "absolute_error")
    if armature_adjusted_worst:
        files.append(f"diagnostics/{armature_adjusted_worst}")
    mass_adapter_worst = write_worst_csv(out_dir / "diagnostics" / "forward_dynamics_mass_adapter_random_activation_worst_qacc_error.csv", mass_adapter_df, "absolute_error")
    if mass_adapter_worst:
        files.append(f"diagnostics/{mass_adapter_worst}")
    residual_worst = write_worst_csv(out_dir / "diagnostics" / "forward_dynamics_generalized_force_residual_worst.csv", residual_df, "absolute_force_residual")
    if residual_worst:
        files.append(f"diagnostics/{residual_worst}")
    rollout_q_worst = write_worst_csv(out_dir / "diagnostics" / "forward_dynamics_rollout_worst_q_error.csv", rollout_state_df, "q_error")
    if rollout_q_worst:
        files.append(f"diagnostics/{rollout_q_worst}")
    rollout_qdot_worst = write_worst_csv(out_dir / "diagnostics" / "forward_dynamics_rollout_worst_qdot_error.csv", rollout_state_df, "qdot_error")
    if rollout_qdot_worst:
        files.append(f"diagnostics/{rollout_qdot_worst}")
    rollout_qacc_worst = write_worst_csv(out_dir / "diagnostics" / "forward_dynamics_rollout_worst_qacc_error.csv", rollout_qacc_df, "absolute_error")
    if rollout_qacc_worst:
        files.append(f"diagnostics/{rollout_qacc_worst}")
    if planned:
        files.append("forward_dynamics_plan.csv")
    finite = _finite_qacc(df)
    random_finite = _finite_qacc(random_df)
    no_armature_finite = _finite_qacc(no_armature_df)
    armature_adjusted_finite = _finite_qacc(armature_adjusted_df)
    mass_adapter_finite = _finite_qacc(mass_adapter_df)
    residual_finite = residual_df[residual_df["status"] == "evaluated"] if not residual_df.empty else residual_df
    rollout_qacc_finite = _finite_qacc(rollout_qacc_df)
    rollout_state_finite = rollout_state_df[rollout_state_df["status"] == "evaluated"] if not rollout_state_df.empty else rollout_state_df
    instant_status = _status(df, mapping)
    random_status = _status(random_df, mapping)
    rollout_qacc_status = _status(rollout_qacc_df, mapping)
    rollout_state_status = _rollout_status(rollout_state_df, mapping)
    status = _combined_status([instant_status, random_status, rollout_qacc_status, rollout_state_status])
    reason = None
    if status == "failed":
        reason = "Forward dynamics mismatch exceeds thresholds; inspect diagnostics/forward_dynamics_*_worst_*.csv."
    elif status == "not evaluated":
        reason = "No directly comparable independent-coordinate acceleration rows."
    return {
        "status": status,
        **({"reason": reason} if reason else {}),
        "planned_experiments": len(planned),
        "evaluated_rows": int(len(finite)) if not finite.empty else 0,
        "max_abs_error": float(finite["absolute_error"].max()) if not finite.empty else None,
        "rmse_abs_error": rmse(finite["absolute_error"]) if not finite.empty else None,
        "random_activation_status": random_status,
        "random_activation_evaluated_rows": int(len(random_finite)) if not random_finite.empty else 0,
        "random_activation_max_abs_error": float(random_finite["absolute_error"].max()) if not random_finite.empty else None,
        "random_activation_rmse_abs_error": rmse(random_finite["absolute_error"]) if not random_finite.empty else None,
        "diagnostic_no_mujoco_armature_max_abs_error": float(no_armature_finite["absolute_error"].max()) if not no_armature_finite.empty else None,
        "diagnostic_no_mujoco_armature_rmse_abs_error": rmse(no_armature_finite["absolute_error"]) if not no_armature_finite.empty else None,
        "diagnostic_sensitivity_no_armature_fro_relative_error": sensitivity_metrics.get("sensitivity_no_armature_fro_relative_error"),
        "diagnostic_sensitivity_armature_adjusted_fro_relative_error": sensitivity_metrics.get("sensitivity_armature_adjusted_fro_relative_error"),
        "diagnostic_armature_adjusted_random_activation_max_abs_error": float(armature_adjusted_finite["absolute_error"].max()) if not armature_adjusted_finite.empty else None,
        "diagnostic_armature_adjusted_random_activation_rmse_abs_error": rmse(armature_adjusted_finite["absolute_error"]) if not armature_adjusted_finite.empty else None,
        "diagnostic_mass_adapter_random_activation_max_abs_error": float(mass_adapter_finite["absolute_error"].max()) if not mass_adapter_finite.empty else None,
        "diagnostic_mass_adapter_random_activation_rmse_abs_error": rmse(mass_adapter_finite["absolute_error"]) if not mass_adapter_finite.empty else None,
        "diagnostic_generalized_force_residual_max_abs": float(residual_finite["absolute_force_residual"].max()) if not residual_finite.empty else None,
        "diagnostic_generalized_force_residual_rmse": rmse(residual_finite["generalized_force_residual"]) if not residual_finite.empty else None,
        "rollout_status": rollout_state_status,
        "rollout_qacc_status": rollout_qacc_status,
        "rollout_evaluated_rows": int(len(rollout_state_finite)) if not rollout_state_finite.empty else 0,
        "rollout_max_q_error": float(rollout_state_finite["q_error"].max()) if not rollout_state_finite.empty else None,
        "rollout_max_qdot_error": float(rollout_state_finite["qdot_error"].max()) if not rollout_state_finite.empty else None,
        "rollout_qacc_max_abs_error": float(rollout_qacc_finite["absolute_error"].max()) if not rollout_qacc_finite.empty else None,
        "rollout_qacc_rmse_abs_error": rmse(rollout_qacc_finite["absolute_error"]) if not rollout_qacc_finite.empty else None,
        "warning_threshold": float(mapping.thresholds.get("forward_dynamics_accel_warning", 5.0)),
        "failure_threshold": float(mapping.thresholds.get("forward_dynamics_accel_fail", 20.0)),
        "files": files,
    }
