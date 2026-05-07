from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from msk_equivalence.checks.kinematics import _pose_values
from msk_equivalence.checks.inverse_dynamics import _coupler_constraints, _dependent_coordinates, _with_coupled_coordinates
from msk_equivalence.checks.muscle_torque import _calibration_values, _interp_activation_for_force, _mujoco_actuator_force, _opensim_actuation
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


def _build_activation_adapter(osim: Any, mjcf: Any, mapping: MappingConfig, opensim_pose: dict[str, float], mujoco_pose: dict[str, float]) -> dict[str, list[tuple[float, float]]]:
    adapter: dict[str, list[tuple[float, float]]] = {}
    for item in mapping.muscles:
        oname = mapping.side_name(item, "opensim")
        mname = mapping.side_name(item, "mujoco")
        if not oname or not mname:
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
    seed = int(experiment.get("seed", 20260507))
    vector_count = int(experiment.get("vector_count", 20))
    active_range = _parse_range(experiment.get("active_muscles_per_vector"), (5, 10))
    names = [mapping.side_name(item, "mujoco") for item in mapping.muscles]
    names = [name for name in names if name]
    rng = np.random.default_rng(seed)
    vectors: list[dict[str, float]] = []
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
    for vector_idx, mujoco_activations in enumerate(_random_activation_vectors(mapping)):
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
        "drift test, not a proof that arbitrary long RL rollouts remain bitwise identical.\n",
    )
    files = [
        "forward_dynamics_smoke_test.csv",
        "forward_dynamics_random_activation_qacc.csv",
        "forward_dynamics_rollout_state_error.csv",
        "forward_dynamics_rollout_qacc_error.csv",
        "forward_dynamics_notes.md",
    ]
    worst = write_worst_csv(out_dir / "diagnostics" / "forward_dynamics_worst_acceleration_error.csv", df, "absolute_error")
    if worst:
        files.append(f"diagnostics/{worst}")
    random_worst = write_worst_csv(out_dir / "diagnostics" / "forward_dynamics_random_activation_worst_qacc_error.csv", random_df, "absolute_error")
    if random_worst:
        files.append(f"diagnostics/{random_worst}")
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
