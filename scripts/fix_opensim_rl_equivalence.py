from __future__ import annotations

import re
from pathlib import Path

from msk_equivalence.loaders.mujoco_loader import MuJoCoModel
from msk_equivalence.mapping import MappingConfig


ROOT = Path(__file__).resolve().parents[1]
OSIM = ROOT / "MimicMSK_Model_opensim" / "MimicMSK_OpenSim.osim"
MJCF = ROOT / "MimicMSK_Model_mujoco" / "body" / "myofullbody.xml"
MAPPING = ROOT / "configs" / "model_mapping.yaml"


def replace_coordinate_default(text: str, name: str, value: str) -> tuple[str, int]:
    pattern = re.compile(
        rf'(<Coordinate name="{re.escape(name)}">.*?<default_value>)(.*?)(</default_value>)',
        re.DOTALL,
    )
    return pattern.subn(rf"\g<1>{value}\g<3>", text, count=1)


def set_coordinate_constraint_free(text: str, name: str) -> tuple[str, int]:
    pattern = re.compile(
        rf'(<Coordinate name="{re.escape(name)}">.*?<is_free_to_satisfy_constraints>)(false|true)(</is_free_to_satisfy_constraints>)',
        re.DOTALL,
    )
    return pattern.subn(r"\g<1>true\g<3>", text, count=1)


def fmt(value: float) -> str:
    return f"{float(value):.12g}"


def replace_body_inertial(
    text: str,
    name: str,
    mass: float,
    mass_center: list[float],
    inertia: list[float],
) -> tuple[str, int]:
    inertia6 = list(inertia[:3]) + [0.0, 0.0, 0.0]
    pattern = re.compile(
        rf'(<Body name="{re.escape(name)}">.*?<mass>)(.*?)(</mass>.*?'
        rf"<mass_center>)(.*?)(</mass_center>.*?<inertia>)(.*?)(</inertia>)",
        re.DOTALL,
    )
    replacement = (
        rf"\g<1>{fmt(mass)}\g<3>"
        + " ".join(fmt(v) for v in mass_center)
        + rf"\g<5>"
        + " ".join(fmt(v) for v in inertia6)
        + rf"\g<7>"
    )
    return pattern.subn(replacement, text, count=1)


def sync_body_inertials(text: str) -> tuple[str, int]:
    mapping = MappingConfig.load(MAPPING)
    mjcf = MuJoCoModel.load(MJCF)
    inertials = {row["name"]: row for row in mjcf.body_inertials()}

    changed = 0
    missing: list[str] = []
    for item in mapping.bodies:
        opensim_name = mapping.side_name(item, "opensim")
        mujoco_name = mapping.side_name(item, "mujoco")
        if not opensim_name or not mujoco_name:
            continue
        row = inertials.get(mujoco_name)
        if row is None:
            missing.append(f"{opensim_name}->{mujoco_name}")
            continue
        text, count = replace_body_inertial(
            text,
            opensim_name,
            float(row["mass"]),
            [float(v) for v in row["com"]],
            [float(v) for v in row["inertia"]],
        )
        if count == 1:
            changed += 1
        else:
            missing.append(f"{opensim_name}->{mujoco_name}")

    if missing:
        raise RuntimeError(f"Failed to sync body inertials: {', '.join(missing[:20])}")
    return text, changed


def deactivate_wrap_objects(text: str) -> tuple[str, int]:
    pattern = re.compile(
        r"(<Wrap(?:Sphere|Cylinder|Torus|Ellipsoid)\b[^>]*>\s*<components />\s*<active>)true(</active>)",
        re.DOTALL,
    )
    return pattern.subn(r"\g<1>false\g<2>", text)


def main() -> None:
    text = OSIM.read_text()

    # Keep the OpenSim default pose usable in the OpenSim GUI. MuJoCo/RL
    # alignment is applied by side-specific pose samples in model_mapping.yaml.
    root_updates = {
        "root_ty": "0.825",
        "root_rx": "-1.5707963",
    }
    for name, value in root_updates.items():
        text, count = replace_coordinate_default(text, name, value)
        if count != 1:
            raise RuntimeError(f"Expected to update one default for {name}, updated {count}")

    dependent_coordinates = sorted(
        {
            match.group(1).strip()
            for match in re.finditer(r"<dependent_coordinate_name>(.*?)</dependent_coordinate_name>", text)
        }
    )
    if not dependent_coordinates:
        raise RuntimeError("No CoordinateCouplerConstraint dependent coordinates found")

    changed = 0
    missing: list[str] = []
    for name in dependent_coordinates:
        text, count = set_coordinate_constraint_free(text, name)
        if count == 1:
            changed += 1
        else:
            missing.append(name)

    if missing:
        raise RuntimeError(f"Failed to update dependent coordinates: {', '.join(missing)}")

    text, inertial_count = sync_body_inertials(text)
    text, wrap_count = deactivate_wrap_objects(text)

    OSIM.write_text(text)
    print(f"updated {OSIM}")
    print("root defaults updated:", ", ".join(root_updates))
    print(f"dependent coordinates marked free-to-satisfy constraints: {changed}")
    print(f"body inertials synchronized from MuJoCo: {inertial_count}")
    print(f"OpenSim wrap objects deactivated to match MuJoCo neutral tendon geometry: {wrap_count}")


if __name__ == "__main__":
    main()
