from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from msk_equivalence.errors import DependencyMissingError, ModelLoadError
from msk_equivalence.utils import vec3


def _opensim():
    try:
        import opensim  # type: ignore
    except ImportError as exc:
        raise DependencyMissingError(
            "OpenSim Python API is not installed. Install OpenSim 4.x Python bindings "
            "and verify `python -c \"import opensim\"` before running OpenSim-backed checks."
        ) from exc
    return opensim


def _set_size(obj: Any) -> int:
    return int(obj.getSize()) if obj is not None and hasattr(obj, "getSize") else 0


def _safe_name(obj: Any) -> str:
    try:
        return str(obj.getName())
    except Exception:
        return ""


def _safe_float(fn: Any, default: float = float("nan")) -> float:
    try:
        return float(fn())
    except Exception:
        return default


@dataclass
class OpenSimModel:
    path: Path
    opensim: Any
    model: Any
    state: Any
    default_coordinates: dict[str, float]

    @classmethod
    def load(cls, path: Path) -> "OpenSimModel":
        if not path.exists():
            raise FileNotFoundError(f"OpenSim model does not exist: {path}")
        osim = _opensim()
        try:
            model = osim.Model(str(path.resolve()))
            state = model.initSystem()
            defaults: dict[str, float] = {}
            coord_set = model.getCoordinateSet()
            for i in range(_set_size(coord_set)):
                coord = coord_set.get(i)
                try:
                    defaults[_safe_name(coord)] = float(coord.getDefaultValue())
                except Exception:
                    defaults[_safe_name(coord)] = 0.0
        except Exception as exc:
            raise ModelLoadError(f"Failed to load OpenSim model {path}: {exc}") from exc
        return cls(path=path, opensim=osim, model=model, state=state, default_coordinates=defaults)

    def _set(self, getter: str) -> Any:
        return getattr(self.model, getter)()

    def names_from_set(self, getter: str) -> list[str]:
        obj_set = self._set(getter)
        return [_safe_name(obj_set.get(i)) for i in range(_set_size(obj_set))]

    @property
    def bodies(self) -> list[str]:
        return self.names_from_set("getBodySet")

    @property
    def coordinates(self) -> list[str]:
        return self.names_from_set("getCoordinateSet")

    @property
    def muscles(self) -> list[str]:
        return self.names_from_set("getMuscles")

    @property
    def markers(self) -> list[str]:
        try:
            return self.names_from_set("getMarkerSet")
        except Exception:
            return []

    @property
    def actuators(self) -> list[str]:
        try:
            return self.names_from_set("getActuators")
        except Exception:
            return []

    def hierarchy(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        try:
            joints = self.model.getJointSet()
            for i in range(_set_size(joints)):
                joint = joints.get(i)
                parent = ""
                child = ""
                try:
                    parent = _safe_name(joint.getParentFrame().findBaseFrame())
                    child = _safe_name(joint.getChildFrame().findBaseFrame())
                except Exception:
                    pass
                rows.append({"joint": _safe_name(joint), "parent": parent, "child": child})
        except Exception:
            return rows
        return rows

    def body_inertials(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        body_set = self.model.getBodySet()
        for i in range(_set_size(body_set)):
            body = body_set.get(i)
            inertia = [float("nan")] * 6
            try:
                inert = body.getInertia()
                inertia = [
                    float(inert.getMoments().get(0)),
                    float(inert.getMoments().get(1)),
                    float(inert.getMoments().get(2)),
                    float(inert.getProducts().get(0)),
                    float(inert.getProducts().get(1)),
                    float(inert.getProducts().get(2)),
                ]
            except Exception:
                pass
            rows.append(
                {
                    "name": _safe_name(body),
                    "mass": _safe_float(body.getMass),
                    "com": vec3(body.getMassCenter()).tolist() if hasattr(body, "getMassCenter") else [np.nan] * 3,
                    "inertia": inertia,
                    "frame": "OpenSim body frame",
                }
            )
        return rows

    def total_mass(self) -> float:
        return _safe_float(lambda: self.model.getTotalMass(self.state))

    def set_pose(self, values: dict[str, float]) -> None:
        coord_set = self.model.getCoordinateSet()
        pose = dict(self.default_coordinates)
        pose.update(values)
        for name, value in pose.items():
            try:
                coord = coord_set.get(name)
                coord.setValue(self.state, float(value), False)
            except Exception:
                continue
        try:
            self.model.realizePosition(self.state)
        except Exception:
            pass

    def body_position(self, name: str) -> np.ndarray:
        body = self.model.getBodySet().get(name)
        return vec3(body.getPositionInGround(self.state))

    def body_rotation_matrix(self, name: str) -> np.ndarray:
        body = self.model.getBodySet().get(name)
        rot = body.getRotationInGround(self.state)
        return np.array([[float(rot.get(i, j)) for j in range(3)] for i in range(3)], dtype=float)

    def marker_position(self, name: str) -> np.ndarray:
        marker = self.model.getMarkerSet().get(name)
        return vec3(marker.getLocationInGround(self.state))

    def whole_body_com(self) -> np.ndarray:
        return vec3(self.model.calcMassCenterPosition(self.state))

    def muscle_length(self, name: str) -> float:
        return float(self.model.getMuscles().get(name).getLength(self.state))

    def moment_arm(self, muscle_name: str, coordinate_name: str) -> float:
        muscle = self.model.getMuscles().get(muscle_name)
        coord = self.model.getCoordinateSet().get(coordinate_name)
        return float(muscle.computeMomentArm(self.state, coord))
