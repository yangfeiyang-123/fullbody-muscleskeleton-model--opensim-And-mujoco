from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read mapping files. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc
    if not path.exists():
        raise FileNotFoundError(f"Mapping file does not exist: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Mapping file must contain a YAML mapping at top level: {path}")
    return data


def _entries(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    raw = data.get(key, [])
    if isinstance(raw, dict):
        return [dict({"opensim": k}, **(v if isinstance(v, dict) else {"mujoco": v})) for k, v in raw.items()]
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _name(item: dict[str, Any], side: str) -> str | None:
    value = item.get(side)
    if isinstance(value, dict):
        return (
            value.get("name")
            or value.get("joint")
            or value.get("qpos")
            or value.get("tendon")
            or value.get("actuator")
            or value.get("geom")
            or value.get("geom1")
            or value.get("geom2")
            or value.get("site")
            or value.get("ground")
        )
    if value is None:
        return None
    return str(value)


@dataclass(frozen=True)
class MappingConfig:
    path: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "MappingConfig":
        return cls(path=path, raw=_load_yaml(path))

    def entries(self, key: str) -> list[dict[str, Any]]:
        return _entries(self.raw, key)

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return self.entries("bodies")

    @property
    def coordinates(self) -> list[dict[str, Any]]:
        return self.entries("coordinates")

    @property
    def markers(self) -> list[dict[str, Any]]:
        return self.entries("markers")

    @property
    def muscles(self) -> list[dict[str, Any]]:
        return self.entries("muscles")

    @property
    def contacts(self) -> list[dict[str, Any]]:
        return self.entries("contacts")

    @property
    def pose_samples(self) -> list[dict[str, Any]]:
        samples = self.raw.get("pose_samples", [{"name": "neutral", "q": {}}])
        if not isinstance(samples, list) or not samples:
            return [{"name": "neutral", "q": {}}]
        return [s if isinstance(s, dict) else {"name": str(s), "q": {}} for s in samples]

    @property
    def conventions(self) -> dict[str, Any]:
        value = self.raw.get("conventions", {})
        return value if isinstance(value, dict) else {}

    @property
    def thresholds(self) -> dict[str, Any]:
        value = self.raw.get("thresholds", {})
        return value if isinstance(value, dict) else {}

    @property
    def dynamics_experiments(self) -> dict[str, Any]:
        value = self.raw.get("dynamics_experiments", {})
        return value if isinstance(value, dict) else {}

    def opensim_names(self, key: str) -> list[str]:
        return [n for n in (_name(item, "opensim") for item in self.entries(key)) if n]

    def mujoco_names(self, key: str) -> list[str]:
        return [n for n in (_name(item, "mujoco") for item in self.entries(key)) if n]

    def coordinate_value(self, item: dict[str, Any], side: str) -> str | None:
        return _name(item, side)

    def side_name(self, item: dict[str, Any], side: str) -> str | None:
        return _name(item, side)
