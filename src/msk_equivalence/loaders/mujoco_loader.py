from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from msk_equivalence.errors import DependencyMissingError, ModelLoadError


def _mujoco():
    try:
        import mujoco  # type: ignore
    except ImportError as exc:
        raise DependencyMissingError(
            "MuJoCo Python binding is not installed. Install it with `pip install mujoco` "
            "and verify `python -c \"import mujoco\"` before running MuJoCo-backed checks."
        ) from exc
    return mujoco


def _name(mj: Any, model: Any, obj_type: int, idx: int) -> str:
    value = mj.mj_id2name(model, obj_type, idx)
    return "" if value is None else str(value)


@dataclass
class MuJoCoModel:
    path: Path
    mujoco: Any
    model: Any
    data: Any

    @classmethod
    def load(cls, path: Path) -> "MuJoCoModel":
        if not path.exists():
            raise FileNotFoundError(f"MuJoCo MJCF model does not exist: {path}")
        mj = _mujoco()
        try:
            model = mj.MjModel.from_xml_path(str(path.resolve()))
            data = mj.MjData(model)
            mj.mj_forward(model, data)
        except Exception as exc:
            raise ModelLoadError(f"Failed to load MuJoCo model {path}: {exc}") from exc
        return cls(path=path, mujoco=mj, model=model, data=data)

    def _names(self, obj_type: int, count: int) -> list[str]:
        return [_name(self.mujoco, self.model, obj_type, i) for i in range(count)]

    @property
    def bodies(self) -> list[str]:
        return [n for n in self._names(self.mujoco.mjtObj.mjOBJ_BODY, self.model.nbody) if n and n != "world"]

    @property
    def joints(self) -> list[str]:
        return [n for n in self._names(self.mujoco.mjtObj.mjOBJ_JOINT, self.model.njnt) if n]

    @property
    def coordinates(self) -> list[str]:
        names = []
        for i in range(self.model.nq):
            names.append(f"qpos[{i}]")
        names.extend(self.joints)
        return names

    @property
    def actuators(self) -> list[str]:
        return [n for n in self._names(self.mujoco.mjtObj.mjOBJ_ACTUATOR, self.model.nu) if n]

    @property
    def muscles(self) -> list[str]:
        tendons = [n for n in self._names(self.mujoco.mjtObj.mjOBJ_TENDON, self.model.ntendon) if n]
        return sorted(set(tendons + self.actuators))

    @property
    def sites(self) -> list[str]:
        return [n for n in self._names(self.mujoco.mjtObj.mjOBJ_SITE, self.model.nsite) if n]

    @property
    def markers(self) -> list[str]:
        return self.sites

    def hierarchy(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for body_id in range(1, self.model.nbody):
            parent_id = int(self.model.body_parentid[body_id])
            rows.append(
                {
                    "joint": "",
                    "parent": _name(self.mujoco, self.model, self.mujoco.mjtObj.mjOBJ_BODY, parent_id),
                    "child": _name(self.mujoco, self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_id),
                }
            )
        return rows

    def body_inertials(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for body_id in range(1, self.model.nbody):
            rows.append(
                {
                    "name": _name(self.mujoco, self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_id),
                    "mass": float(self.model.body_mass[body_id]),
                    "com": np.array(self.model.body_ipos[body_id], dtype=float).tolist(),
                    "inertia": np.array(self.model.body_inertia[body_id], dtype=float).tolist(),
                    "frame": "MuJoCo inertial frame/body-aligned principal axes",
                }
            )
        return rows

    def total_mass(self) -> float:
        return float(np.sum(self.model.body_mass[1:]))

    def _joint_id(self, name: str) -> int | None:
        obj = self.mujoco.mjtObj.mjOBJ_JOINT
        idx = self.mujoco.mj_name2id(self.model, obj, name)
        return None if idx < 0 else int(idx)

    def qpos_index(self, spec: str) -> int | None:
        if spec.startswith("qpos[") and spec.endswith("]"):
            return int(spec[5:-1])
        joint_id = self._joint_id(spec)
        if joint_id is None:
            return None
        return int(self.model.jnt_qposadr[joint_id])

    def set_pose(self, values: dict[str, float]) -> None:
        if hasattr(self.model, "qpos0"):
            self.data.qpos[:] = self.model.qpos0
        else:
            self.data.qpos[:] = 0
        for name, value in values.items():
            idx = self.qpos_index(name)
            if idx is not None and 0 <= idx < self.model.nq:
                self.data.qpos[idx] = float(value)
        self.mujoco.mj_forward(self.model, self.data)

    def body_position(self, name: str) -> np.ndarray:
        body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise KeyError(name)
        return np.array(self.data.xpos[body_id], dtype=float)

    def body_rotation_matrix(self, name: str) -> np.ndarray:
        body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise KeyError(name)
        return np.array(self.data.xmat[body_id], dtype=float).reshape(3, 3)

    def site_position(self, name: str) -> np.ndarray:
        site_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise KeyError(name)
        return np.array(self.data.site_xpos[site_id], dtype=float)

    def marker_position(self, name: str) -> np.ndarray:
        return self.site_position(name)

    def whole_body_com(self) -> np.ndarray:
        masses = np.array(self.model.body_mass[1:], dtype=float)
        if masses.size == 0 or np.sum(masses) <= 0:
            return np.full(3, np.nan)
        positions = np.array(self.data.xipos[1:], dtype=float)
        return np.average(positions, axis=0, weights=masses)

    def tendon_length(self, name: str) -> float:
        tendon_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_TENDON, name)
        if tendon_id >= 0:
            return float(self.data.ten_length[tendon_id])
        actuator_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id >= 0 and hasattr(self.data, "actuator_length"):
            return float(self.data.actuator_length[actuator_id])
        raise KeyError(name)

    def moment_arm_numeric(self, tendon_name: str, qpos_name: str, eps: float = 1e-6) -> float:
        idx = self.qpos_index(qpos_name)
        if idx is None:
            raise KeyError(qpos_name)
        base = float(self.data.qpos[idx])
        self.data.qpos[idx] = base + eps
        self.mujoco.mj_forward(self.model, self.data)
        lp = self.tendon_length(tendon_name)
        self.data.qpos[idx] = base - eps
        self.mujoco.mj_forward(self.model, self.data)
        lm = self.tendon_length(tendon_name)
        self.data.qpos[idx] = base
        self.mujoco.mj_forward(self.model, self.data)
        return -float((lp - lm) / (2.0 * eps))
