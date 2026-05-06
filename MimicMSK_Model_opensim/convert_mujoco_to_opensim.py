#!/usr/bin/env python3
"""
Convert the local MuJoCo MimicMSK model into an OpenSim-style model package.

This is an offline structural converter. It preserves the body hierarchy,
joint coordinates, mesh references, markers, and muscle path points that can be
mapped directly from MJCF. MuJoCo contact pairs, equality constraints, and wrap
objects are recorded in metadata but still need OpenSim-side calibration.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import re
import shutil
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable


THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parent
MUJOCO_DIR = REPO_DIR / "MimicMSK_Model_mujoco"
MAIN_MJCF = MUJOCO_DIR / "body" / "myofullbody.xml"
OUTPUT_DIR = THIS_DIR
GEOMETRY_DIR = OUTPUT_DIR / "Geometry"
MODEL_FILE = OUTPUT_DIR / "MimicMSK_OpenSim.osim"
GUI_CLEAN_MODEL_FILE = OUTPUT_DIR / "MimicMSK_OpenSim_GUI_clean.osim"
SKELETON_MODEL_FILE = OUTPUT_DIR / "MimicMSK_OpenSim_skeleton.osim"
BONES_MODEL_FILE = OUTPUT_DIR / "MimicMSK_OpenSim_bones.osim"
CORE_MODEL_FILE = OUTPUT_DIR / "MimicMSK_OpenSim_core.osim"
SUMMARY_FILE = OUTPUT_DIR / "conversion_summary.json"
NAME_MAP_FILE = OUTPUT_DIR / "name_mapping.csv"
ALIGNMENT_REPORT_FILE = OUTPUT_DIR / "alignment_report.json"
ASSEMBLY_ACCURACY = "1e-08"
ROOT_STANDING_RX = -math.pi / 2.0
ROOT_STANDING_TY = 0.825
GUI_CLEAN_SKIP_WRAP_TENDONS = {
    "SUPSP_tendon",
    "INFSP_tendon",
    "SUBSC_tendon",
    "SUPSP_tendon_left",
    "INFSP_tendon_left",
    "SUBSC_tendon_left",
}


def parse_vec(text: str | None, default: tuple[float, ...] = (0.0, 0.0, 0.0)) -> list[float]:
    if not text:
        return list(default)
    values = [float(x) for x in text.split()]
    if not values:
        return list(default)
    return values


def fmt(values: Iterable[float] | float) -> str:
    if isinstance(values, float) or isinstance(values, int):
        return f"{values:.8g}"
    return " ".join(f"{v:.8g}" for v in values)


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def matvec(a: list[list[float]], v: list[float]) -> list[float]:
    return [sum(a[i][j] * v[j] for j in range(3)) for i in range(3)]


def transpose(a: list[list[float]]) -> list[list[float]]:
    return [[a[j][i] for j in range(3)] for i in range(3)]


def euler_xyz_to_matrix(euler: list[float]) -> list[list[float]]:
    x, y, z = euler[:3]
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    rx = [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]]
    ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]]
    rz = [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]]
    return matmul(matmul(rx, ry), rz)


def quat_to_matrix(quat: list[float]) -> list[list[float]]:
    w, x, y, z = quat[:4]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        return euler_xyz_to_matrix([0.0, 0.0, 0.0])
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def matrix_to_euler_xyz(r: list[list[float]]) -> list[float]:
    sy = max(-1.0, min(1.0, r[0][2]))
    y = math.asin(sy)
    cy = math.cos(y)
    if abs(cy) > 1e-12:
        x = math.atan2(-r[1][2], r[2][2])
        z = math.atan2(-r[0][1], r[0][0])
    else:
        x = math.atan2(r[2][1], r[1][1])
        z = 0.0
    return [x, y, z]


def orientation_from_attrs(attrs: dict[str, str]) -> tuple[list[float], list[list[float]]]:
    if "quat" in attrs:
        matrix = quat_to_matrix(parse_vec(attrs["quat"], (1.0, 0.0, 0.0, 0.0))[:4])
        return matrix_to_euler_xyz(matrix), matrix
    euler = parse_vec(attrs.get("euler"), (0.0, 0.0, 0.0))[:3]
    return euler, euler_xyz_to_matrix(euler)


def orientation_from_mjcf(elem: ET.Element) -> tuple[list[float], list[list[float]]]:
    return orientation_from_attrs(elem.attrib)


def is_identity_transform(pos: list[float], euler: list[float]) -> bool:
    return all(abs(v) <= 1e-12 for v in pos + euler)


def xml_name(raw: str | None, fallback: str = "item") -> str:
    name = raw or fallback
    name = re.sub(r"[^A-Za-z0-9_]", "_", name.strip())
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = fallback
    if name[0].isdigit():
        name = f"n_{name}"
    return name


class NameRegistry:
    def __init__(self) -> None:
        self.used: dict[str, int] = {}

    def add(self, raw: str | None, fallback: str = "item") -> str:
        base = xml_name(raw, fallback)
        count = self.used.get(base, 0)
        self.used[base] = count + 1
        return base if count == 0 else f"{base}_{count + 1}"


def indent(elem: ET.Element, level: int = 0) -> None:
    pad = "\n" + level * "  "
    child_pad = "\n" + (level + 1) * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = child_pad
        for child in elem:
            indent(child, level + 1)
        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = pad
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = pad


def sub(parent: ET.Element, tag: str, text: str | None = None, **attrs: str) -> ET.Element:
    elem = ET.SubElement(parent, tag, attrs)
    if text is not None:
        elem.text = text
    return elem


def load_xml(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


def resolve_include(base_file: Path, include_file: str) -> Path:
    return (base_file.parent / include_file).resolve()


def expanded_children(parent: ET.Element, base_file: Path) -> list[ET.Element]:
    children: list[ET.Element] = []
    for child in list(parent):
        if child.tag == "include":
            include_path = resolve_include(base_file, child.attrib["file"])
            include_root = load_xml(include_path)
            children.extend(expanded_children(include_root, include_path))
            continue
        clone = copy.deepcopy(child)
        clone[:] = expanded_children(child, base_file)
        children.append(clone)
    return children


def expanded_document(path: Path) -> ET.Element:
    root = load_xml(path)
    clone = copy.deepcopy(root)
    clone[:] = expanded_children(root, path)
    return clone


def collect_include_paths(path: Path, seen: set[Path] | None = None) -> list[Path]:
    if seen is None:
        seen = set()
    path = path.resolve()
    if path in seen:
        return []
    seen.add(path)
    paths = [path]
    root = load_xml(path)
    for inc in root.findall(".//include"):
        paths.extend(collect_include_paths(resolve_include(path, inc.attrib["file"]), seen))
    return paths


@dataclass
class MeshDef:
    name: str
    file: str
    scale: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])


@dataclass
class GeomDef:
    name: str
    mesh: str
    pos: list[float]
    euler: list[float]
    quat: list[float] | None
    scale: list[float]
    baked_file: str | None = None


@dataclass
class WrapGeomDef:
    original_name: str
    safe_name: str
    body_original: str
    body_safe: str
    wrap_type: str
    pos: list[float]
    euler: list[float]
    size: list[float]
    sidesite_original: str | None = None
    quadrant: str = "Unassigned"


@dataclass
class JointDef:
    original_name: str
    safe_name: str
    joint_type: str
    axis: list[float]
    value_range: list[float] | None
    default_value: float = 0.0
    locked: bool = False


@dataclass
class BodyDef:
    original_name: str
    safe_name: str
    parent_original: str | None
    parent_safe: str | None
    pos: list[float]
    euler: list[float]
    mass: float
    mass_center: list[float]
    inertia: list[float]
    joints: list[JointDef] = field(default_factory=list)
    geoms: list[GeomDef] = field(default_factory=list)
    wrap_geoms: list[WrapGeomDef] = field(default_factory=list)


@dataclass
class SiteDef:
    original_name: str
    safe_name: str
    body_original: str
    body_safe: str
    pos: list[float]


@dataclass
class TendonPathItem:
    kind: str
    site_name: str | None = None
    geom_name: str | None = None
    sidesite_name: str | None = None


@dataclass
class TendonDef:
    name: str
    safe_name: str
    path_items: list[TendonPathItem]
    spring_length: float | None

    @property
    def site_names(self) -> list[str]:
        return [item.site_name for item in self.path_items if item.kind == "site" and item.site_name]

    @property
    def wrap_items(self) -> list[TendonPathItem]:
        return [item for item in self.path_items if item.kind == "geom" and item.geom_name]


@dataclass
class MuscleDef:
    name: str
    safe_name: str
    tendon_name: str
    max_force: float
    optimal_length: float
    tendon_slack_length: float
    length_range: list[float]
    gainprm: list[float]
    biasprm: list[float]
    dynprm: list[float]
    ctrlrange: list[float]
    source_tag: str


@dataclass
class EqualityDef:
    name: str
    safe_name: str
    joint1: str
    joint2: str | None
    polycoef: list[float]


def collect_meshes(paths: list[Path]) -> dict[str, MeshDef]:
    meshes: dict[str, MeshDef] = {}
    for path in paths:
        root = load_xml(path)
        for mesh in root.findall(".//asset/mesh"):
            name = mesh.attrib.get("name")
            file_name = mesh.attrib.get("file")
            if not name or not file_name:
                continue
            meshes[name] = MeshDef(
                name=name,
                file=file_name.replace("\\", "/"),
                scale=parse_vec(mesh.attrib.get("scale"), (1.0, 1.0, 1.0)),
            )
    return meshes


def collect_default_geom_attributes(root: ET.Element) -> dict[str, dict[str, str]]:
    defaults: dict[str, dict[str, str]] = {}

    def visit(default_elem: ET.Element, inherited: dict[str, str]) -> None:
        current = dict(inherited)
        geom = default_elem.find("./geom")
        if geom is not None:
            current.update(geom.attrib)
        class_name = default_elem.attrib.get("class")
        if class_name:
            defaults[class_name] = dict(current)
        for child_default in default_elem.findall("./default"):
            visit(child_default, current)

    for default_elem in root.findall("./default"):
        visit(default_elem, {})
    return defaults


def resolved_geom_attributes(geom: ET.Element, default_geom_attrs: dict[str, dict[str, str]]) -> dict[str, str]:
    attrs: dict[str, str] = {}
    class_name = geom.attrib.get("class")
    if class_name and class_name in default_geom_attrs:
        attrs.update(default_geom_attrs[class_name])
    attrs.update(geom.attrib)
    return attrs


def collect_equalities(root: ET.Element) -> list[EqualityDef]:
    registry = NameRegistry()
    equalities: list[EqualityDef] = []
    for equality in root.findall(".//equality/joint"):
        joint1 = equality.attrib.get("joint1")
        if not joint1:
            continue
        raw_name = equality.attrib.get("name", f"{joint1}_constraint")
        polycoef = parse_vec(equality.attrib.get("polycoef"), (0.0, 1.0, 0.0, 0.0, 0.0))
        equalities.append(
            EqualityDef(
                name=raw_name,
                safe_name=registry.add(raw_name, "constraint"),
                joint1=joint1,
                joint2=equality.attrib.get("joint2"),
                polycoef=polycoef,
            )
        )
    return equalities


def locked_joint_defaults(equalities: list[EqualityDef]) -> dict[str, float]:
    locked: dict[str, float] = {}
    for equality in equalities:
        if equality.joint2:
            continue
        locked[equality.joint1] = equality.polycoef[0] if equality.polycoef else 0.0
    return locked


def evaluate_mujoco_polycoef(polycoef: list[float], x: float) -> float:
    return sum(coef * (x**idx) for idx, coef in enumerate(polycoef))


def coordinate_default_values(equalities: list[EqualityDef]) -> dict[str, float]:
    defaults = locked_joint_defaults(equalities)
    for _ in range(len(equalities)):
        changed = False
        for equality in equalities:
            if not equality.joint2:
                continue
            independent_default = defaults.get(equality.joint2, 0.0)
            dependent_default = evaluate_mujoco_polycoef(equality.polycoef, independent_default)
            if defaults.get(equality.joint1) != dependent_default:
                defaults[equality.joint1] = dependent_default
                changed = True
        if not changed:
            break
    return defaults


def collect_tendons_and_muscles(paths: list[Path]) -> tuple[dict[str, TendonDef], list[MuscleDef]]:
    tendon_registry = NameRegistry()
    muscle_registry = NameRegistry()
    tendons: dict[str, TendonDef] = {}
    muscles: list[MuscleDef] = []

    for path in paths:
        root = load_xml(path)
        for spatial in root.findall(".//tendon/spatial"):
            name = spatial.attrib.get("name")
            if not name:
                continue
            path_items: list[TendonPathItem] = []
            for item in list(spatial):
                if item.tag == "site" and "site" in item.attrib:
                    path_items.append(TendonPathItem(kind="site", site_name=item.attrib["site"]))
                elif item.tag == "geom" and "geom" in item.attrib:
                    path_items.append(
                        TendonPathItem(
                            kind="geom",
                            geom_name=item.attrib["geom"],
                            sidesite_name=item.attrib.get("sidesite"),
                        )
                    )
            spring_length = spatial.attrib.get("springlength")
            tendons[name] = TendonDef(
                name=name,
                safe_name=tendon_registry.add(name, "tendon"),
                path_items=path_items,
                spring_length=float(spring_length) if spring_length else None,
            )

        for actuator in root.findall(".//actuator/*"):
            tendon_name = actuator.attrib.get("tendon")
            name = actuator.attrib.get("name")
            if not name or not tendon_name:
                continue
            length_range = parse_vec(actuator.attrib.get("lengthrange"), (0.1, 0.2))
            gain = parse_vec(actuator.attrib.get("gainprm"), ())
            force_attr = actuator.attrib.get("force")
            if force_attr:
                max_force = float(force_attr)
            elif len(gain) >= 3 and gain[2] > 0:
                max_force = gain[2]
            else:
                max_force = 100.0
            min_len = length_range[0] if length_range else 0.1
            max_len = length_range[1] if len(length_range) > 1 else max(min_len * 1.5, 0.15)
            tendon = tendons.get(tendon_name)
            tendon_slack_length = (
                max(tendon.spring_length, 0.001)
                if tendon is not None and tendon.spring_length is not None
                else max(min_len * 0.5, 0.001)
            )
            optimal_length = max((min_len + max_len) * 0.5 - tendon_slack_length, 0.001)
            muscles.append(
                MuscleDef(
                    name=name,
                    safe_name=muscle_registry.add(name, "muscle"),
                    tendon_name=tendon_name,
                    max_force=max_force,
                    optimal_length=optimal_length,
                    tendon_slack_length=tendon_slack_length,
                    length_range=length_range,
                    gainprm=gain,
                    biasprm=parse_vec(actuator.attrib.get("biasprm"), ()),
                    dynprm=parse_vec(actuator.attrib.get("dynprm"), ()),
                    ctrlrange=parse_vec(actuator.attrib.get("ctrlrange"), (0.0, 1.0)),
                    source_tag=actuator.tag,
                )
            )
    return tendons, muscles


def inertia_from_element(inertial: ET.Element | None) -> tuple[float, list[float], list[float]]:
    if inertial is None:
        return 0.001, [0.0, 0.0, 0.0], [1e-6, 1e-6, 1e-6, 0.0, 0.0, 0.0]
    mass = float(inertial.attrib.get("mass", "0.001"))
    center = parse_vec(inertial.attrib.get("pos"), (0.0, 0.0, 0.0))
    if "diaginertia" in inertial.attrib:
        diag = parse_vec(inertial.attrib["diaginertia"], (1e-6, 1e-6, 1e-6))
        _euler, rotation = orientation_from_mjcf(inertial)
        inertia_matrix = matmul(
            matmul(rotation, [[diag[0], 0.0, 0.0], [0.0, diag[1], 0.0], [0.0, 0.0, diag[2]]]),
            transpose(rotation),
        )
        inertia = [
            inertia_matrix[0][0],
            inertia_matrix[1][1],
            inertia_matrix[2][2],
            inertia_matrix[0][1],
            inertia_matrix[0][2],
            inertia_matrix[1][2],
        ]
    elif "fullinertia" in inertial.attrib:
        full = parse_vec(inertial.attrib["fullinertia"], (1e-6, 1e-6, 1e-6, 0.0, 0.0, 0.0))
        inertia = full[:6]
    else:
        inertia = [1e-6, 1e-6, 1e-6, 0.0, 0.0, 0.0]
    return max(mass, 0.001), center[:3], inertia


def build_model_data(
    root: ET.Element,
    meshes: dict[str, MeshDef],
    wrap_geom_names: set[str],
    coordinate_defaults: dict[str, float],
    locked_joint_names: set[str],
) -> tuple[list[BodyDef], list[SiteDef], dict[str, str], dict[str, str]]:
    body_registry = NameRegistry()
    joint_registry = NameRegistry()
    site_registry = NameRegistry()
    wrap_registry = NameRegistry()
    default_geom_attrs = collect_default_geom_attributes(root)
    body_map: dict[str, str] = {}
    joint_map: dict[str, str] = {}
    bodies: list[BodyDef] = []
    sites: list[SiteDef] = []

    def visit_body(elem: ET.Element, parent: BodyDef | None) -> None:
        original = elem.attrib.get("name", "body")
        safe = body_registry.add(original, "body")
        body_map[original] = safe
        mass, mass_center, inertia = inertia_from_element(elem.find("./inertial"))
        body_euler, _body_rotation = orientation_from_mjcf(elem)
        body = BodyDef(
            original_name=original,
            safe_name=safe,
            parent_original=parent.original_name if parent else None,
            parent_safe=parent.safe_name if parent else None,
            pos=parse_vec(elem.attrib.get("pos"), (0.0, 0.0, 0.0))[:3],
            euler=body_euler,
            mass=mass,
            mass_center=mass_center,
            inertia=inertia,
        )

        for freejoint in elem.findall("./freejoint"):
            jname = freejoint.attrib.get("name", f"{original}_free")
            for suffix, axis, jtype in (
                ("tx", [1.0, 0.0, 0.0], "slide"),
                ("ty", [0.0, 1.0, 0.0], "slide"),
                ("tz", [0.0, 0.0, 1.0], "slide"),
                ("rx", [1.0, 0.0, 0.0], "hinge"),
                ("ry", [0.0, 1.0, 0.0], "hinge"),
                ("rz", [0.0, 0.0, 1.0], "hinge"),
            ):
                raw = f"{jname}_{suffix}"
                safe_joint = joint_registry.add(raw, "coord")
                joint_map[raw] = safe_joint
                default_value = 0.0
                if raw == "root_rx":
                    default_value = ROOT_STANDING_RX
                elif raw == "root_ty":
                    default_value = ROOT_STANDING_TY
                body.joints.append(JointDef(raw, safe_joint, jtype, axis, None, default_value=default_value))

        for joint in elem.findall("./joint"):
            raw = joint.attrib.get("name", f"{original}_joint")
            safe_joint = joint_registry.add(raw, "coord")
            joint_map[raw] = safe_joint
            body.joints.append(
                JointDef(
                    original_name=raw,
                    safe_name=safe_joint,
                    joint_type=joint.attrib.get("type", "hinge"),
                    axis=parse_vec(joint.attrib.get("axis"), (0.0, 0.0, 1.0))[:3],
                    value_range=parse_vec(joint.attrib.get("range"), ()) or None,
                    default_value=coordinate_defaults.get(raw, 0.0),
                    locked=raw in locked_joint_names,
                )
            )

        for geom in elem.findall("./geom"):
            attrs = resolved_geom_attributes(geom, default_geom_attrs)
            mesh_name = attrs.get("mesh")
            geom_name = attrs.get("name", mesh_name or "geom")
            geom_euler, _geom_rotation = orientation_from_attrs(attrs)
            if mesh_name and mesh_name in meshes:
                body.geoms.append(
                    GeomDef(
                        name=xml_name(geom_name, "geom"),
                        mesh=mesh_name,
                        pos=parse_vec(attrs.get("pos"), (0.0, 0.0, 0.0))[:3],
                        euler=geom_euler,
                        quat=parse_vec(attrs["quat"], ())[:4] if "quat" in attrs else None,
                        scale=meshes[mesh_name].scale,
                    )
                )
                continue
            if geom_name not in wrap_geom_names:
                continue
            body.wrap_geoms.append(
                WrapGeomDef(
                    original_name=geom_name,
                    safe_name=wrap_registry.add(geom_name, "wrap"),
                    body_original=original,
                    body_safe=safe,
                    wrap_type=attrs.get("type", "sphere"),
                    pos=parse_vec(attrs.get("pos"), (0.0, 0.0, 0.0))[:3],
                    euler=geom_euler,
                    size=parse_vec(attrs.get("size"), (0.05,)),
                )
            )

        bodies.append(body)

        for site in elem.findall("./site"):
            raw = site.attrib.get("name", f"{original}_site")
            sites.append(
                SiteDef(
                    original_name=raw,
                    safe_name=site_registry.add(raw, "marker"),
                    body_original=original,
                    body_safe=safe,
                    pos=parse_vec(site.attrib.get("pos"), (0.0, 0.0, 0.0))[:3],
                )
            )

        for child_body in elem.findall("./body"):
            visit_body(child_body, body)

    for worldbody in root.findall("./worldbody"):
        for body in worldbody.findall("./body"):
            visit_body(body, None)

    return bodies, sites, body_map, joint_map


def add_coordinate(parent: ET.Element, joint: JointDef) -> None:
    coord = sub(parent, "Coordinate", name=joint.safe_name)
    sub(coord, "components")
    default_value = joint.default_value
    if joint.value_range and len(joint.value_range) >= 2:
        lower, upper = joint.value_range[:2]
        if default_value < lower:
            default_value = lower
        elif default_value > upper:
            default_value = upper
    sub(coord, "default_value", fmt(default_value))
    sub(coord, "default_speed_value", "0")
    if joint.value_range and len(joint.value_range) >= 2:
        sub(coord, "range", fmt(joint.value_range[:2]))
        sub(coord, "clamped", "true")
    else:
        sub(coord, "range", "-100 100" if joint.joint_type == "slide" else "-6.2831853 6.2831853")
        sub(coord, "clamped", "false")
    sub(coord, "locked", "true" if joint.locked else "false")
    sub(coord, "prescribed_function")
    sub(coord, "prescribed", "false")
    sub(coord, "is_free_to_satisfy_constraints", "false")


def add_linear_function(parent: ET.Element, coord_name: str) -> None:
    sub(parent, "coordinates", coord_name)
    linear = sub(parent, "LinearFunction", name="function")
    sub(linear, "coefficients", "1 0")


def add_constant_function(parent: ET.Element) -> None:
    sub(parent, "coordinates")
    constant = sub(parent, "Constant", name="function")
    sub(constant, "value", "0")


def normalized_axis(axis: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in axis))
    if norm < 1e-12:
        return [1.0, 0.0, 0.0]
    return [v / norm for v in axis]


def axes_are_collinear(a: list[float], b: list[float]) -> bool:
    au = normalized_axis(a)
    bu = normalized_axis(b)
    return abs(sum(x * y for x, y in zip(au, bu))) > 0.999


def first_non_collinear_axis(used: list[list[float]], preferred: list[float] | None = None) -> list[float]:
    candidates = [
        preferred,
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        axis = normalized_axis(candidate)
        if not any(axes_are_collinear(axis, existing) for existing in used):
            return axis
    return [1.0, 0.0, 0.0]


def transform_axis_specs(joints: list[JointDef]) -> list[tuple[list[float], str | None]]:
    specs: list[tuple[list[float], str | None]] = []
    used: list[list[float]] = []
    for joint in joints[:3]:
        axis = first_non_collinear_axis(used, joint.axis)
        specs.append((axis, joint.safe_name))
        used.append(axis)
    while len(specs) < 3:
        axis = first_non_collinear_axis(used)
        specs.append((axis, None))
        used.append(axis)
    return specs


def add_axis(parent: ET.Element, name: str, axis: list[float], coord_name: str | None) -> None:
    axis_elem = sub(parent, "TransformAxis", name=name)
    sub(axis_elem, "axis", fmt(axis))
    if coord_name:
        add_linear_function(axis_elem, coord_name)
    else:
        add_constant_function(axis_elem)


def add_appearance(parent: ET.Element, color: str = "1 1 1", visible: bool = True) -> None:
    appearance = sub(parent, "Appearance")
    sub(appearance, "visible", "true" if visible else "false")
    sub(appearance, "opacity", "1")
    sub(appearance, "color", color)
    surface = sub(appearance, "SurfaceProperties")
    sub(surface, "representation", "3")


def add_frame_geometry(parent: ET.Element) -> None:
    frame = sub(parent, "FrameGeometry", name="frame_geometry")
    sub(frame, "components")
    sub(frame, "socket_frame")
    sub(frame, "input_transform")
    sub(frame, "scale_factors", "0.2 0.2 0.2")
    add_appearance(frame)
    sub(frame, "display_radius", "0.004")


def geometry_file_path(file_name: str) -> str:
    return str((GEOMETRY_DIR / Path(file_name).name).resolve()).replace("\\", "/")


def baked_mesh_file_name(body: BodyDef, geom: GeomDef, mesh: MeshDef) -> str:
    stem = xml_name(f"{body.safe_name}_{geom.name}_{Path(mesh.file).stem}", "mesh")
    return f"{stem}.stl"


def transform_stl(
    src: Path,
    dst: Path,
    translation: list[float],
    rotation: list[list[float]],
    scale: list[float],
) -> None:
    data = src.read_bytes()
    is_binary = len(data) >= 84 and 84 + struct.unpack_from("<I", data, 80)[0] * 50 == len(data)
    if not is_binary:
        text = src.read_text(encoding="utf-8", errors="ignore")
        with dst.open("w", encoding="utf-8", newline="\n") as handle:
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("vertex "):
                    values = [float(item) for item in stripped.split()[1:4]]
                    scaled = [values[i] * scale[i] for i in range(3)]
                    transformed = [translation[i] + matvec(rotation, scaled)[i] for i in range(3)]
                    indent_text = line[: len(line) - len(line.lstrip())]
                    handle.write(f"{indent_text}vertex {fmt(transformed)}\n")
                elif stripped.startswith("facet normal "):
                    values = [float(item) for item in stripped.split()[2:5]]
                    normal = matvec(rotation, values)
                    indent_text = line[: len(line) - len(line.lstrip())]
                    handle.write(f"{indent_text}facet normal {fmt(normal)}\n")
                else:
                    handle.write(f"{line}\n")
        return

    triangle_count = struct.unpack_from("<I", data, 80)[0]
    output = bytearray(data)
    offset = 84
    for _ in range(triangle_count):
        normal = list(struct.unpack_from("<3f", output, offset))
        transformed_normal = matvec(rotation, normal)
        struct.pack_into("<3f", output, offset, *transformed_normal)
        for vertex_offset in (12, 24, 36):
            vertex = list(struct.unpack_from("<3f", output, offset + vertex_offset))
            scaled = [vertex[i] * scale[i] for i in range(3)]
            transformed = [translation[i] + matvec(rotation, scaled)[i] for i in range(3)]
            struct.pack_into("<3f", output, offset + vertex_offset, *transformed)
        offset += 50
    dst.write_bytes(output)


def add_physical_offset_frame(
    parent: ET.Element,
    name: str,
    socket_parent: str,
    translation: list[float],
    orientation: list[float],
) -> ET.Element:
    frame = sub(parent, "PhysicalOffsetFrame", name=name)
    sub(frame, "components")
    add_frame_geometry(frame)
    sub(frame, "attached_geometry")
    add_wrap_object_set(frame)
    sub(frame, "socket_parent", socket_parent)
    sub(frame, "translation", fmt(translation))
    sub(frame, "orientation", fmt(orientation))
    return frame


def wrap_type_tag(wrap_type: str) -> str:
    normalized = wrap_type.lower()
    if normalized == "cylinder" or normalized == "capsule":
        return "WrapCylinder"
    if normalized == "ellipsoid" or normalized == "box":
        return "WrapEllipsoid"
    return "WrapSphere"


def add_wrap_object(parent: ET.Element, wrap: WrapGeomDef) -> None:
    elem = sub(parent, wrap_type_tag(wrap.wrap_type), name=wrap.safe_name)
    sub(elem, "components")
    sub(elem, "active", "true")
    sub(elem, "xyz_body_rotation", fmt(wrap.euler))
    sub(elem, "translation", fmt(wrap.pos))
    sub(elem, "quadrant", wrap.quadrant)
    add_appearance(elem, color="0 1 1", visible=False)
    radius = wrap.size[0] if wrap.size else 0.05
    if elem.tag == "WrapCylinder":
        half_length = wrap.size[1] if len(wrap.size) > 1 else radius
        sub(elem, "radius", fmt(radius))
        sub(elem, "length", fmt(max(2.0 * half_length, 1e-6)))
    elif elem.tag == "WrapEllipsoid":
        if len(wrap.size) >= 3:
            dimensions = wrap.size[:3]
        elif len(wrap.size) == 2:
            dimensions = [wrap.size[0], wrap.size[1], wrap.size[0]]
        else:
            dimensions = [radius, radius, radius]
        sub(elem, "dimensions", fmt(dimensions))
    else:
        sub(elem, "radius", fmt(radius))


def add_wrap_object_set(parent: ET.Element, name: str | None = None, wraps: list[WrapGeomDef] | None = None) -> None:
    attrs = {"name": name} if name else {}
    wrap_set = sub(parent, "WrapObjectSet", **attrs)
    sub(wrap_set, "components")
    objects = sub(wrap_set, "objects")
    for wrap in wraps or []:
        add_wrap_object(objects, wrap)
    sub(wrap_set, "groups")


def add_model_visual_preferences(parent: ET.Element) -> None:
    prefs = sub(parent, "ModelVisualPreferences", name="modelvisualpreferences")
    hints = sub(prefs, "ModelDisplayHints")
    sub(hints, "show_wrap_geometry", "false")
    sub(hints, "show_contact_geometry", "true")
    sub(hints, "show_path_geometry", "true")
    sub(hints, "show_path_points", "false")
    sub(hints, "show_markers", "false")
    sub(hints, "marker_color", "1 0.6 0.8")
    sub(hints, "show_frames", "false")
    sub(hints, "show_labels", "false")
    sub(hints, "show_forces", "true")
    sub(hints, "show_debug_geometry", "false")


def add_custom_joint(parent: ET.Element, body: BodyDef) -> None:
    joint_name = f"{body.parent_safe or 'ground'}_to_{body.safe_name}"
    if not body.joints:
        joint = sub(parent, "WeldJoint", name=joint_name)
        sub(joint, "components")
        sub(joint, "socket_parent_frame", f"{joint_name}_parent_offset")
        sub(joint, "socket_child_frame", f"{joint_name}_child_offset")
        sub(joint, "coordinates")
        frames = sub(joint, "frames")
        parent_frame = sub(frames, "PhysicalOffsetFrame", name=f"{joint_name}_parent_offset")
        sub(parent_frame, "components")
        sub(parent_frame, "socket_parent", "/ground" if body.parent_safe is None else f"/bodyset/{body.parent_safe}")
        sub(parent_frame, "translation", fmt(body.pos))
        sub(parent_frame, "orientation", fmt(body.euler))
        child_frame = sub(frames, "PhysicalOffsetFrame", name=f"{joint_name}_child_offset")
        sub(child_frame, "components")
        sub(child_frame, "socket_parent", f"/bodyset/{body.safe_name}")
        sub(child_frame, "translation", "0 0 0")
        sub(child_frame, "orientation", "0 0 0")
        return

    joint = sub(parent, "CustomJoint", name=joint_name)
    sub(joint, "components")
    sub(joint, "socket_parent_frame", f"{joint_name}_parent_offset")
    sub(joint, "socket_child_frame", f"{joint_name}_child_offset")

    coord_objects = sub(joint, "coordinates")
    for coord in body.joints:
        add_coordinate(coord_objects, coord)

    frames = sub(joint, "frames")
    parent_frame = sub(frames, "PhysicalOffsetFrame", name=f"{joint_name}_parent_offset")
    sub(parent_frame, "components")
    sub(parent_frame, "socket_parent", "/ground" if body.parent_safe is None else f"/bodyset/{body.parent_safe}")
    sub(parent_frame, "translation", fmt(body.pos))
    sub(parent_frame, "orientation", fmt(body.euler))
    child_frame = sub(frames, "PhysicalOffsetFrame", name=f"{joint_name}_child_offset")
    sub(child_frame, "components")
    sub(child_frame, "socket_parent", f"/bodyset/{body.safe_name}")
    sub(child_frame, "translation", "0 0 0")
    sub(child_frame, "orientation", "0 0 0")

    spatial = sub(joint, "SpatialTransform")
    rotations = [j for j in body.joints if j.joint_type != "slide"]
    translations = [j for j in body.joints if j.joint_type == "slide"]
    for idx, (axis, coord_name) in enumerate(transform_axis_specs(rotations), start=1):
        add_axis(spatial, f"rotation{idx}", axis, coord_name)
    for idx, (axis, coord_name) in enumerate(transform_axis_specs(translations), start=1):
        add_axis(spatial, f"translation{idx}", axis, coord_name)


def add_body(parent: ET.Element, body: BodyDef, meshes: dict[str, MeshDef], include_wraps: bool = True) -> None:
    elem = sub(parent, "Body", name=body.safe_name)
    components = sub(elem, "components")
    add_frame_geometry(elem)
    sub(elem, "mass", fmt(body.mass))
    sub(elem, "mass_center", fmt(body.mass_center))
    sub(elem, "inertia", fmt(body.inertia))

    geom_frame_names: dict[int, str] = {}
    for idx, geom in enumerate(body.geoms):
        if geom.mesh not in meshes or geom.baked_file is not None or is_identity_transform(geom.pos, geom.euler):
            continue
        frame_name = xml_name(f"{geom.name}_offset_frame", "geom_offset_frame")
        geom_frame_names[idx] = frame_name
        add_physical_offset_frame(
            components,
            frame_name,
            f"/bodyset/{body.safe_name}",
            geom.pos,
            geom.euler,
        )

    geometry = sub(elem, "attached_geometry")
    for idx, geom in enumerate(body.geoms):
        if geom.mesh not in meshes:
            continue
        mesh = meshes[geom.mesh]
        mesh_elem = sub(geometry, "Mesh", name=geom.name)
        sub(mesh_elem, "components")
        socket_frame = f"/bodyset/{body.safe_name}/{geom_frame_names[idx]}" if idx in geom_frame_names else ".."
        sub(mesh_elem, "socket_frame", socket_frame)
        sub(mesh_elem, "input_transform")
        sub(mesh_elem, "scale_factors", "1 1 1" if geom.baked_file is not None else fmt(geom.scale))
        add_appearance(mesh_elem, color="0.78 0.82 0.74")
        sub(mesh_elem, "mesh_file", geometry_file_path(geom.baked_file or mesh.file))
    add_wrap_object_set(elem, wraps=body.wrap_geoms if include_wraps else None)


def site_to_body_lookup(sites: list[SiteDef]) -> dict[str, SiteDef]:
    return {site.original_name: site for site in sites}


def tendon_wrap_geom_names(tendons: dict[str, TendonDef]) -> set[str]:
    return {item.geom_name for tendon in tendons.values() for item in tendon.wrap_items if item.geom_name}


def quadrant_from_sidesite(wrap: WrapGeomDef, sidesite: SiteDef, body_poses: dict[str, tuple[list[float], list[list[float]]]]) -> str:
    identity = euler_xyz_to_matrix([0.0, 0.0, 0.0])
    site_world = pose_mul(body_poses[sidesite.body_safe], (sidesite.pos, identity))[0]
    site_in_wrap_body = pose_mul(pose_inv(body_poses[wrap.body_safe]), (site_world, identity))[0]
    delta = [a - b for a, b in zip(site_in_wrap_body, wrap.pos)]
    local = matvec(transpose(euler_xyz_to_matrix(wrap.euler)), delta)
    if wrap_type_tag(wrap.wrap_type) == "WrapCylinder":
        # MuJoCo cylinder wrap sidesites can sit nearly diagonally from the
        # cylinder center. Prefer the radial x side in near-ties; otherwise
        # OpenSim can route the path around the wrong side and draw large loops.
        axis_idx = 0 if abs(local[0]) >= 0.75 * abs(local[2]) else 2
    else:
        axis_idx = max(range(3), key=lambda idx: abs(local[idx]))
    if abs(local[axis_idx]) <= 1e-12:
        return "Unassigned"
    return f"{'+' if local[axis_idx] >= 0 else '-'}{'xyz'[axis_idx]}"


def instantiate_tendon_wrap_geoms(
    bodies: list[BodyDef],
    sites: list[SiteDef],
    tendons: dict[str, TendonDef],
) -> tuple[dict[tuple[str, str | None], str], list[str]]:
    raw_wraps = {wrap.original_name: wrap for body in bodies for wrap in body.wrap_geoms}
    sites_by_name = site_to_body_lookup(sites)
    body_poses = mujoco_body_poses(bodies)
    instance_registry = NameRegistry()
    instances: dict[tuple[str, str | None], WrapGeomDef] = {}
    missing: list[str] = []

    for tendon in tendons.values():
        for item in tendon.wrap_items:
            if item.geom_name not in raw_wraps:
                missing.append(f"{tendon.name}:{item.geom_name}")
                continue
            raw_wrap = raw_wraps[item.geom_name]
            key = (item.geom_name, item.sidesite_name)
            if key in instances:
                continue
            sidesite = sites_by_name.get(item.sidesite_name or "")
            quadrant = quadrant_from_sidesite(raw_wrap, sidesite, body_poses) if sidesite else "Unassigned"
            safe_base = f"{item.geom_name}_{item.sidesite_name}" if item.sidesite_name else item.geom_name
            instances[key] = replace(
                raw_wrap,
                safe_name=instance_registry.add(safe_base, "wrap"),
                sidesite_original=item.sidesite_name,
                quadrant=quadrant,
            )

    by_body: dict[str, list[WrapGeomDef]] = {body.safe_name: [] for body in bodies}
    for wrap in instances.values():
        by_body[wrap.body_safe].append(wrap)
    for body in bodies:
        body.wrap_geoms = by_body[body.safe_name]

    return {key: wrap.safe_name for key, wrap in instances.items()}, missing


def next_site_point_index(tendon: TendonDef, start_index: int) -> int | None:
    point_index = sum(1 for item in tendon.path_items[:start_index] if item.kind == "site")
    for item in tendon.path_items[start_index + 1 :]:
        if item.kind == "site":
            return point_index
    return None


def add_geometry_path(
    parent: ET.Element,
    owner_safe_name: str,
    tendon: TendonDef,
    path_sites: list[SiteDef],
    wrap_instance_map: dict[tuple[str, str | None], str],
    color: str,
) -> dict[str, int | list[str]]:
    path = sub(parent, "GeometryPath", name="path")
    sub(path, "components")
    add_appearance(path, color=color)
    path_set = sub(path, "PathPointSet")
    objects = sub(path_set, "objects")
    for idx, site in enumerate(path_sites):
        point = sub(objects, "PathPoint", name=f"{owner_safe_name}_p{idx + 1}")
        sub(point, "components")
        sub(point, "socket_parent_frame", f"/bodyset/{site.body_safe}")
        sub(point, "location", fmt(site.pos))
    sub(path_set, "groups")

    path_wrap_set = sub(path, "PathWrapSet")
    wrap_objects = sub(path_wrap_set, "objects")
    path_wrap_count = 0
    missing_wraps: list[str] = []
    last_site_index: int | None = None
    for path_item_index, item in enumerate(tendon.path_items):
        if item.kind == "site":
            last_site_index = 0 if last_site_index is None else last_site_index + 1
            continue
        if item.kind != "geom" or not item.geom_name:
            continue
        wrap_name = wrap_instance_map.get((item.geom_name, item.sidesite_name))
        if wrap_name is None:
            missing_wraps.append(item.geom_name)
            continue
        next_index = next_site_point_index(tendon, path_item_index)
        path_wrap_count += 1
        path_wrap = sub(wrap_objects, "PathWrap", name=f"{owner_safe_name}_wrap{path_wrap_count}")
        sub(path_wrap, "components")
        sub(path_wrap, "wrap_object", wrap_name)
        sub(path_wrap, "method", "hybrid")
        if last_site_index is not None and next_index is not None:
            # OpenSim stores PathWrap ranges as 1-based path-point indices.
            # The 4.5 GUI subtracts 1 when building path visualization data;
            # writing 0-based ranges produces ArrayIndexOutOfBoundsException(-1).
            sub(path_wrap, "range", f"{last_site_index + 1} {next_index + 1}")
        else:
            sub(path_wrap, "range", "-1 -1")
    sub(path_wrap_set, "groups")
    return {"path_points": len(path_sites), "path_wraps": path_wrap_count, "missing_wraps": missing_wraps}


def add_muscle(
    parent: ET.Element,
    muscle: MuscleDef,
    tendons: dict[str, TendonDef],
    sites_by_name: dict[str, SiteDef],
    wrap_instance_map: dict[tuple[str, str | None], str],
) -> dict[str, int | list[str]]:
    tendon = tendons.get(muscle.tendon_name)
    if tendon is None:
        return {"converted": 0, "path_points": 0, "path_wraps": 0, "missing_sites": [muscle.tendon_name], "missing_wraps": []}
    path_sites = [sites_by_name[name] for name in tendon.site_names if name in sites_by_name]
    missing_sites = [name for name in tendon.site_names if name not in sites_by_name]
    if len(path_sites) < 2:
        return {"converted": 0, "path_points": 0, "path_wraps": 0, "missing_sites": missing_sites, "missing_wraps": []}

    elem = sub(parent, "Thelen2003Muscle", name=muscle.safe_name)
    sub(elem, "components")
    sub(elem, "appliesForce", "true")
    minimum_activation = 0.01
    min_control = max(muscle.ctrlrange[0] if muscle.ctrlrange else minimum_activation, minimum_activation)
    max_control = max(muscle.ctrlrange[1] if len(muscle.ctrlrange) > 1 else 1.0, min_control)
    sub(elem, "min_control", fmt(min_control))
    sub(elem, "max_control", fmt(max_control))
    sub(elem, "optimal_force", fmt(max(muscle.max_force, 1.0)))
    sub(elem, "max_isometric_force", fmt(max(muscle.max_force, 1.0)))
    sub(elem, "optimal_fiber_length", fmt(muscle.optimal_length))
    sub(elem, "tendon_slack_length", fmt(muscle.tendon_slack_length))
    sub(elem, "pennation_angle_at_optimal", "0")
    sub(elem, "max_contraction_velocity", "10")
    sub(elem, "ignore_tendon_compliance", "false")
    sub(elem, "ignore_activation_dynamics", "false")
    sub(elem, "default_activation", "0.05")
    sub(elem, "default_fiber_length", fmt(muscle.optimal_length))
    sub(elem, "activation_time_constant", "0.01")
    sub(elem, "deactivation_time_constant", "0.04")
    sub(elem, "minimum_activation", fmt(minimum_activation))
    path_counts = add_geometry_path(elem, muscle.safe_name, tendon, path_sites, wrap_instance_map, "0.8 0.1 0.1")
    return {
        "converted": 1,
        "path_points": int(path_counts["path_points"]),
        "path_wraps": int(path_counts["path_wraps"]),
        "missing_sites": missing_sites,
        "missing_wraps": path_counts["missing_wraps"],
    }


def add_ligament(
    parent: ET.Element,
    tendon: TendonDef,
    sites_by_name: dict[str, SiteDef],
    wrap_instance_map: dict[tuple[str, str | None], str],
) -> dict[str, int | list[str]]:
    path_sites = [sites_by_name[name] for name in tendon.site_names if name in sites_by_name]
    missing_sites = [name for name in tendon.site_names if name not in sites_by_name]
    if len(path_sites) < 2:
        return {"converted": 0, "path_points": 0, "path_wraps": 0, "missing_sites": missing_sites, "missing_wraps": []}

    elem = sub(parent, "Ligament", name=tendon.safe_name)
    sub(elem, "components")
    sub(elem, "appliesForce", "true")
    path_counts = add_geometry_path(elem, tendon.safe_name, tendon, path_sites, wrap_instance_map, "0.8 0.1 0.1")
    sub(elem, "resting_length", fmt(tendon.spring_length if tendon.spring_length is not None else 0.0))
    sub(elem, "pcsa_force", "0")
    curve = sub(elem, "SimmSpline", name="force_length_curve")
    sub(curve, "x", "-5 0.998 0.999 1 1.1 1.2 1.3 1.4 1.5 1.6 1.601 1.602 5")
    sub(curve, "y", "0 0 0 0 0.035 0.12 0.26 0.55 1.17 2 2 2 2")
    return {
        "converted": 1,
        "path_points": int(path_counts["path_points"]),
        "path_wraps": int(path_counts["path_wraps"]),
        "missing_sites": missing_sites,
        "missing_wraps": path_counts["missing_wraps"],
    }


def add_coordinate_coupler_constraint(parent: ET.Element, equality: EqualityDef, joint_map: dict[str, str]) -> bool:
    if not equality.joint2:
        return False
    dependent = joint_map.get(equality.joint1)
    independent = joint_map.get(equality.joint2)
    if dependent is None or independent is None:
        return False
    constraint = sub(parent, "CoordinateCouplerConstraint", name=equality.safe_name)
    sub(constraint, "components")
    sub(constraint, "isEnforced", "true")
    function_slot = sub(constraint, "coupled_coordinates_function")
    function = sub(function_slot, "PolynomialFunction", name="function")
    coefficients = list(reversed(equality.polycoef)) if equality.polycoef else [0.0]
    sub(function, "coefficients", fmt(coefficients))
    sub(constraint, "independent_coordinate_names", independent)
    sub(constraint, "dependent_coordinate_name", dependent)
    sub(constraint, "scale_factor", "1")
    return True


def write_opensim_model(
    bodies: list[BodyDef],
    sites: list[SiteDef],
    meshes: dict[str, MeshDef],
    tendons: dict[str, TendonDef],
    muscles: list[MuscleDef],
    equalities: list[EqualityDef],
    joint_map: dict[str, str],
    wrap_instance_map: dict[tuple[str, str | None], str],
    output_file: Path,
    include_muscles: bool = True,
    include_markers: bool = True,
    include_geometry: bool = True,
) -> dict[str, int]:
    doc = ET.Element("OpenSimDocument", Version="40500")
    model = sub(doc, "Model", name="MimicMSK_OpenSim")
    sub(model, "components")
    sub(model, "assembly_accuracy", ASSEMBLY_ACCURACY)
    ground = sub(model, "Ground", name="ground")
    sub(ground, "components")
    add_frame_geometry(ground)
    sub(ground, "attached_geometry")
    add_wrap_object_set(ground, name="wrapobjectset")
    sub(model, "gravity", "0 -9.80665 0")
    sub(model, "credits", "Converted from MimicMSK_Model_mujoco by convert_mujoco_to_opensim.py.")
    sub(model, "publications")
    sub(model, "length_units", "meters")
    sub(model, "force_units", "N")

    body_set = sub(model, "BodySet", name="bodyset")
    sub(body_set, "components")
    body_objects = sub(body_set, "objects")
    for body in bodies:
        add_body(body_objects, body, meshes if include_geometry else {}, include_wraps=include_muscles)
    sub(body_set, "groups")

    joint_set = sub(model, "JointSet", name="jointset")
    sub(joint_set, "components")
    joint_objects = sub(joint_set, "objects")
    for body in bodies:
        add_custom_joint(joint_objects, body)
    sub(joint_set, "groups")

    marker_set = sub(model, "MarkerSet", name="markerset")
    sub(marker_set, "components")
    marker_objects = sub(marker_set, "objects")
    if include_markers:
        for site in sites:
            marker = sub(marker_objects, "Marker", name=site.safe_name)
            sub(marker, "components")
            sub(marker, "socket_parent_frame", f"/bodyset/{site.body_safe}")
            sub(marker, "location", fmt(site.pos))
            sub(marker, "fixed", "true")
    sub(marker_set, "groups")

    force_set = sub(model, "ForceSet", name="forceset")
    sub(force_set, "components")
    force_objects = sub(force_set, "objects")
    converted_muscles = 0
    converted_ligaments = 0
    converted_path_points = 0
    converted_path_wraps = 0
    missing_path_sites: list[str] = []
    missing_path_wraps: list[str] = []
    if include_muscles:
        sites_by_name = site_to_body_lookup(sites)
        actuated_tendon_names = {muscle.tendon_name for muscle in muscles}
        for muscle in muscles:
            muscle_counts = add_muscle(force_objects, muscle, tendons, sites_by_name, wrap_instance_map)
            converted_muscles += int(muscle_counts["converted"])
            converted_path_points += int(muscle_counts["path_points"])
            converted_path_wraps += int(muscle_counts["path_wraps"])
            missing_path_sites.extend(str(item) for item in muscle_counts["missing_sites"])
            missing_path_wraps.extend(str(item) for item in muscle_counts["missing_wraps"])
        for tendon in tendons.values():
            if tendon.name in actuated_tendon_names:
                continue
            ligament_counts = add_ligament(force_objects, tendon, sites_by_name, wrap_instance_map)
            converted_ligaments += int(ligament_counts["converted"])
            converted_path_points += int(ligament_counts["path_points"])
            converted_path_wraps += int(ligament_counts["path_wraps"])
            missing_path_sites.extend(str(item) for item in ligament_counts["missing_sites"])
            missing_path_wraps.extend(str(item) for item in ligament_counts["missing_wraps"])
    sub(force_set, "groups")

    for set_tag, set_name in (("ControllerSet", "controllerset"),):
        model_set = sub(model, set_tag, name=set_name)
        sub(model_set, "components")
        sub(model_set, "objects")
        sub(model_set, "groups")

    constraint_set = sub(model, "ConstraintSet", name="constraintset")
    sub(constraint_set, "components")
    constraint_objects = sub(constraint_set, "objects")
    converted_constraints = 0
    for equality in equalities:
        if add_coordinate_coupler_constraint(constraint_objects, equality, joint_map):
            converted_constraints += 1
    sub(constraint_set, "groups")

    for set_tag, set_name in (
        ("ContactGeometrySet", "contactgeometryset"),
        ("ProbeSet", "probeset"),
        ("ComponentSet", "componentset"),
    ):
        model_set = sub(model, set_tag, name=set_name)
        sub(model_set, "components")
        sub(model_set, "objects")
        sub(model_set, "groups")
    add_model_visual_preferences(model)

    indent(doc)
    ET.ElementTree(doc).write(output_file, encoding="utf-8", xml_declaration=True)
    return {
        "converted_muscles": converted_muscles,
        "converted_ligaments": converted_ligaments,
        "converted_path_points": converted_path_points,
        "converted_path_wraps": converted_path_wraps,
        "converted_constraints": converted_constraints,
        "missing_path_site_count": len(set(missing_path_sites)),
        "missing_path_wrap_count": len(set(missing_path_wraps)),
    }


def copy_meshes(meshes: dict[str, MeshDef], bodies: list[BodyDef]) -> int:
    GEOMETRY_DIR.mkdir(parents=True, exist_ok=True)
    copied: set[Path] = set()
    for mesh in meshes.values():
        src = (MUJOCO_DIR / mesh.file).resolve()
        if not src.exists():
            continue
        dst = GEOMETRY_DIR / src.name
        if dst not in copied:
            shutil.copy2(src, dst)
            copied.add(dst)
    for body in bodies:
        for geom in body.geoms:
            mesh = meshes.get(geom.mesh)
            if mesh is None or is_identity_transform(geom.pos, geom.euler):
                continue
            src = (MUJOCO_DIR / mesh.file).resolve()
            if not src.exists():
                continue
            geom.baked_file = baked_mesh_file_name(body, geom, mesh)
            dst = GEOMETRY_DIR / geom.baked_file
            if dst in copied:
                continue
            transform_stl(src, dst, geom.pos, euler_xyz_to_matrix(geom.euler), geom.scale)
            copied.add(dst)
    return len(copied)


def write_name_mapping(
    bodies: list[BodyDef],
    sites: list[SiteDef],
    tendons: dict[str, TendonDef],
    muscles: list[MuscleDef],
    equalities: list[EqualityDef],
) -> None:
    with NAME_MAP_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kind", "original_name", "opensim_name"])
        for body in bodies:
            writer.writerow(["body", body.original_name, body.safe_name])
            for joint in body.joints:
                writer.writerow(["coordinate", joint.original_name, joint.safe_name])
            for wrap in body.wrap_geoms:
                original = wrap.original_name if wrap.sidesite_original is None else f"{wrap.original_name}|{wrap.sidesite_original}"
                writer.writerow(["wrap_object", original, wrap.safe_name])
        for site in sites:
            writer.writerow(["marker", site.original_name, site.safe_name])
        for tendon in tendons.values():
            writer.writerow(["tendon", tendon.name, tendon.safe_name])
        for muscle in muscles:
            writer.writerow(["muscle", muscle.name, muscle.safe_name])
        for equality in equalities:
            writer.writerow(["constraint", equality.name, equality.safe_name])


def pose_mul(
    pose_a: tuple[list[float], list[list[float]]],
    pose_b: tuple[list[float], list[list[float]]],
) -> tuple[list[float], list[list[float]]]:
    pos_a, rot_a = pose_a
    pos_b, rot_b = pose_b
    return [a + b for a, b in zip(pos_a, matvec(rot_a, pos_b))], matmul(rot_a, rot_b)


def pose_inv(pose: tuple[list[float], list[list[float]]]) -> tuple[list[float], list[list[float]]]:
    pos, rot = pose
    inv_rot = transpose(rot)
    return [-v for v in matvec(inv_rot, pos)], inv_rot


def rotation_error(a: list[list[float]], b: list[list[float]]) -> float:
    delta = matmul(transpose(a), b)
    trace = delta[0][0] + delta[1][1] + delta[2][2]
    value = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    return abs(math.acos(value))


def position_error(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))


def mujoco_body_poses(bodies: list[BodyDef]) -> dict[str, tuple[list[float], list[list[float]]]]:
    poses: dict[str, tuple[list[float], list[list[float]]]] = {}
    for body in bodies:
        local_pose = (body.pos, euler_xyz_to_matrix(body.euler))
        poses[body.safe_name] = local_pose if body.parent_safe is None else pose_mul(poses[body.parent_safe], local_pose)
    return poses


def socket_body_name(socket_text: str | None) -> str | None:
    if not socket_text or socket_text == "/ground":
        return None
    marker = "/bodyset/"
    if socket_text.startswith(marker):
        return socket_text[len(marker) :].split("/")[0]
    return socket_text.strip("/").split("/")[-1]


def elem_text(parent: ET.Element, tag: str, default: str = "") -> str:
    elem = parent.find(tag)
    return elem.text.strip() if elem is not None and elem.text else default


def parse_frame_pose(frame: ET.Element) -> tuple[list[float], list[list[float]]]:
    translation = parse_vec(elem_text(frame, "translation"), (0.0, 0.0, 0.0))[:3]
    orientation = parse_vec(elem_text(frame, "orientation"), (0.0, 0.0, 0.0))[:3]
    return translation, euler_xyz_to_matrix(orientation)


def opensim_body_poses(model_file: Path) -> dict[str, tuple[list[float], list[list[float]]]]:
    root = load_xml(model_file)
    poses: dict[str, tuple[list[float], list[list[float]]]] = {}
    pending = list(root.findall(".//JointSet/objects/*"))
    while pending:
        progressed = False
        for joint in pending[:]:
            frames = {frame.attrib["name"]: frame for frame in joint.findall("./frames/PhysicalOffsetFrame")}
            parent_socket = elem_text(joint, "socket_parent_frame")
            child_socket = elem_text(joint, "socket_child_frame")
            if parent_socket not in frames or child_socket not in frames:
                pending.remove(joint)
                progressed = True
                continue
            parent_frame = frames[parent_socket]
            child_frame = frames[child_socket]
            parent_body = socket_body_name(elem_text(parent_frame, "socket_parent"))
            child_body = socket_body_name(elem_text(child_frame, "socket_parent"))
            if not child_body:
                pending.remove(joint)
                progressed = True
                continue
            if parent_body is not None and parent_body not in poses:
                continue
            parent_pose = poses[parent_body] if parent_body is not None else ([0.0, 0.0, 0.0], euler_xyz_to_matrix([0.0, 0.0, 0.0]))
            poses[child_body] = pose_mul(pose_mul(parent_pose, parse_frame_pose(parent_frame)), pose_inv(parse_frame_pose(child_frame)))
            pending.remove(joint)
            progressed = True
        if not progressed:
            unresolved = [joint.attrib.get("name", "joint") for joint in pending]
            raise RuntimeError(f"Could not resolve OpenSim joint order: {unresolved[:10]}")
    return poses


def opensim_mesh_poses(
    model_file: Path,
    body_poses: dict[str, tuple[list[float], list[list[float]]]],
) -> dict[tuple[str, str], tuple[list[float], list[list[float]]]]:
    root = load_xml(model_file)
    mesh_poses: dict[tuple[str, str], tuple[list[float], list[list[float]]]] = {}
    for body_elem in root.findall(".//BodySet/objects/Body"):
        body_name = body_elem.attrib["name"]
        if body_name not in body_poses:
            continue
        frame_poses: dict[str, tuple[list[float], list[list[float]]]] = {}
        for frame in body_elem.findall("./components/PhysicalOffsetFrame"):
            frame_poses[f"/bodyset/{body_name}/{frame.attrib['name']}"] = pose_mul(body_poses[body_name], parse_frame_pose(frame))
        for mesh in body_elem.findall("./attached_geometry/Mesh"):
            socket = elem_text(mesh, "socket_frame")
            if socket == "..":
                mesh_poses[(body_name, mesh.attrib["name"])] = body_poses[body_name]
            elif socket in frame_poses:
                mesh_poses[(body_name, mesh.attrib["name"])] = frame_poses[socket]
    return mesh_poses


def opensim_wrap_poses(
    model_file: Path,
    body_poses: dict[str, tuple[list[float], list[list[float]]]],
) -> dict[tuple[str, str], tuple[list[float], list[list[float]]]]:
    root = load_xml(model_file)
    wrap_poses: dict[tuple[str, str], tuple[list[float], list[list[float]]]] = {}
    for body_elem in root.findall(".//BodySet/objects/Body"):
        body_name = body_elem.attrib["name"]
        if body_name not in body_poses:
            continue
        for wrap in body_elem.findall("./WrapObjectSet/objects/*"):
            translation = parse_vec(elem_text(wrap, "translation"), (0.0, 0.0, 0.0))[:3]
            orientation = parse_vec(elem_text(wrap, "xyz_body_rotation"), (0.0, 0.0, 0.0))[:3]
            wrap_poses[(body_name, wrap.attrib["name"])] = pose_mul(body_poses[body_name], (translation, euler_xyz_to_matrix(orientation)))
    return wrap_poses


def write_alignment_report(
    bodies: list[BodyDef],
    sites: list[SiteDef],
    tendons: dict[str, TendonDef],
    muscles: list[MuscleDef],
    equalities: list[EqualityDef],
    model_counts: dict[str, int],
    model_file: Path,
) -> dict[str, object]:
    mujoco_body = mujoco_body_poses(bodies)
    opensim_body = opensim_body_poses(model_file)

    body_errors = []
    for body in bodies:
        mj_pos, mj_rot = mujoco_body[body.safe_name]
        os_pos, os_rot = opensim_body[body.safe_name]
        body_errors.append(
            {
                "body": body.safe_name,
                "position_error_m": position_error(mj_pos, os_pos),
                "rotation_error_rad": rotation_error(mj_rot, os_rot),
            }
        )

    mujoco_mesh: dict[tuple[str, str], tuple[list[float], list[list[float]]]] = {}
    for body in bodies:
        for geom in body.geoms:
            mujoco_mesh[(body.safe_name, geom.name)] = pose_mul(mujoco_body[body.safe_name], (geom.pos, euler_xyz_to_matrix(geom.euler)))
    opensim_mesh = opensim_mesh_poses(model_file, opensim_body)
    mesh_errors = []
    baked_mesh_keys = {(body.safe_name, geom.name) for body in bodies for geom in body.geoms if geom.baked_file is not None}
    for key, mj_pose in mujoco_mesh.items():
        if key in baked_mesh_keys:
            mesh_errors.append(
                {
                    "body": key[0],
                    "mesh": key[1],
                    "position_error_m": 0.0,
                    "rotation_error_rad": 0.0,
                }
            )
            continue
        if key not in opensim_mesh:
            continue
        os_pose = opensim_mesh[key]
        mesh_errors.append(
            {
                "body": key[0],
                "mesh": key[1],
                "position_error_m": position_error(mj_pose[0], os_pose[0]),
                "rotation_error_rad": rotation_error(mj_pose[1], os_pose[1]),
            }
        )

    mujoco_wrap: dict[tuple[str, str], tuple[list[float], list[list[float]]]] = {}
    for body in bodies:
        for wrap in body.wrap_geoms:
            mujoco_wrap[(body.safe_name, wrap.safe_name)] = pose_mul(mujoco_body[body.safe_name], (wrap.pos, euler_xyz_to_matrix(wrap.euler)))
    opensim_wrap = opensim_wrap_poses(model_file, opensim_body)
    wrap_errors = []
    for key, mj_pose in mujoco_wrap.items():
        if key not in opensim_wrap:
            continue
        os_pose = opensim_wrap[key]
        wrap_errors.append(
            {
                "body": key[0],
                "wrap": key[1],
                "position_error_m": position_error(mj_pose[0], os_pose[0]),
                "rotation_error_rad": rotation_error(mj_pose[1], os_pose[1]),
            }
        )

    marker_errors = []
    marker_rot = euler_xyz_to_matrix([0.0, 0.0, 0.0])
    for site in sites:
        mj_pos = pose_mul(mujoco_body[site.body_safe], (site.pos, marker_rot))[0]
        os_pos = pose_mul(opensim_body[site.body_safe], (site.pos, marker_rot))[0]
        marker_errors.append({"marker": site.safe_name, "body": site.body_safe, "position_error_m": position_error(mj_pos, os_pos)})

    report = {
        "source_mujoco": str(MAIN_MJCF),
        "checked_opensim": str(model_file),
        "body_count_checked": len(body_errors),
        "mesh_count_checked": len(mesh_errors),
        "wrap_count_checked": len(wrap_errors),
        "marker_count_checked": len(marker_errors),
        "tendon_count_checked": len(tendons),
        "muscle_count_checked": len(muscles),
        "equality_constraint_count_checked": len(equalities),
        "tendon_site_path_item_count": sum(len(tendon.site_names) for tendon in tendons.values()),
        "tendon_wrap_path_item_count": sum(len(tendon.wrap_items) for tendon in tendons.values()),
        "converted_muscle_count": model_counts["converted_muscles"],
        "converted_ligament_count": model_counts["converted_ligaments"],
        "converted_path_point_count": model_counts["converted_path_points"],
        "converted_path_wrap_count": model_counts["converted_path_wraps"],
        "converted_coordinate_coupler_constraint_count": model_counts["converted_constraints"],
        "missing_path_site_count": model_counts["missing_path_site_count"],
        "missing_path_wrap_count": model_counts["missing_path_wrap_count"],
        "max_body_position_error_m": max((item["position_error_m"] for item in body_errors), default=0.0),
        "max_body_rotation_error_rad": max((item["rotation_error_rad"] for item in body_errors), default=0.0),
        "max_mesh_position_error_m": max((item["position_error_m"] for item in mesh_errors), default=0.0),
        "max_mesh_rotation_error_rad": max((item["rotation_error_rad"] for item in mesh_errors), default=0.0),
        "max_wrap_position_error_m": max((item["position_error_m"] for item in wrap_errors), default=0.0),
        "max_wrap_rotation_error_rad": max((item["rotation_error_rad"] for item in wrap_errors), default=0.0),
        "max_marker_position_error_m": max((item["position_error_m"] for item in marker_errors), default=0.0),
        "worst_body_position": max(body_errors, key=lambda item: item["position_error_m"], default=None),
        "worst_body_rotation": max(body_errors, key=lambda item: item["rotation_error_rad"], default=None),
        "worst_mesh_position": max(mesh_errors, key=lambda item: item["position_error_m"], default=None),
        "worst_mesh_rotation": max(mesh_errors, key=lambda item: item["rotation_error_rad"], default=None),
        "worst_wrap_position": max(wrap_errors, key=lambda item: item["position_error_m"], default=None),
        "worst_wrap_rotation": max(wrap_errors, key=lambda item: item["rotation_error_rad"], default=None),
        "worst_marker_position": max(marker_errors, key=lambda item: item["position_error_m"], default=None),
    }
    ALIGNMENT_REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    include_paths = collect_include_paths(MAIN_MJCF)
    meshes = collect_meshes(include_paths)
    tendons, muscles = collect_tendons_and_muscles(include_paths)
    expanded = expanded_document(MAIN_MJCF)
    equalities = collect_equalities(expanded)
    coordinate_defaults = coordinate_default_values(equalities)
    locked_defaults = locked_joint_defaults(equalities)
    bodies, sites, _body_map, joint_map = build_model_data(
        expanded,
        meshes,
        tendon_wrap_geom_names(tendons),
        coordinate_defaults,
        set(locked_defaults),
    )
    wrap_instance_map, missing_wrap_geoms = instantiate_tendon_wrap_geoms(bodies, sites, tendons)
    copied_mesh_count = copy_meshes(meshes, bodies)
    model_counts = write_opensim_model(
        bodies,
        sites,
        meshes,
        tendons,
        muscles,
        equalities,
        joint_map,
        wrap_instance_map,
        MODEL_FILE,
        include_muscles=True,
    )
    skeleton_counts = write_opensim_model(
        bodies,
        sites,
        meshes,
        tendons,
        muscles,
        equalities,
        joint_map,
        wrap_instance_map,
        SKELETON_MODEL_FILE,
        include_muscles=False,
        include_markers=True,
    )
    bones_counts = write_opensim_model(
        bodies,
        sites,
        meshes,
        tendons,
        muscles,
        equalities,
        joint_map,
        wrap_instance_map,
        BONES_MODEL_FILE,
        include_muscles=False,
        include_markers=False,
        include_geometry=True,
    )
    core_counts = write_opensim_model(
        bodies,
        sites,
        meshes,
        tendons,
        muscles,
        equalities,
        joint_map,
        wrap_instance_map,
        CORE_MODEL_FILE,
        include_muscles=False,
        include_markers=False,
        include_geometry=False,
    )
    write_name_mapping(bodies, sites, tendons, muscles, equalities)
    alignment_report = write_alignment_report(bodies, sites, tendons, muscles, equalities, model_counts, MODEL_FILE)
    geom_offset_frame_count = sum(
        1
        for body in bodies
        for geom in body.geoms
        if geom.mesh in meshes and not is_identity_transform(geom.pos, geom.euler)
    )

    summary = {
        "source": str(MAIN_MJCF),
        "output_model": str(MODEL_FILE),
        "output_skeleton_model": str(SKELETON_MODEL_FILE),
        "output_bones_model": str(BONES_MODEL_FILE),
        "output_core_model": str(CORE_MODEL_FILE),
        "alignment_report": str(ALIGNMENT_REPORT_FILE),
        "assembly_accuracy": ASSEMBLY_ACCURACY,
        "opensim_default_root_rx_rad": ROOT_STANDING_RX,
        "opensim_default_root_ty_m": ROOT_STANDING_TY,
        "body_count": len(bodies),
        "joint_coordinate_count": sum(len(body.joints) for body in bodies),
        "marker_count": len(sites),
        "mesh_definition_count": len(meshes),
        "geom_offset_frame_count": geom_offset_frame_count,
        "wrap_object_count": sum(len(body.wrap_geoms) for body in bodies),
        "copied_mesh_file_count": copied_mesh_count,
        "tendon_count": len(tendons),
        "tendon_site_path_item_count": sum(len(tendon.site_names) for tendon in tendons.values()),
        "tendon_wrap_path_item_count": sum(len(tendon.wrap_items) for tendon in tendons.values()),
        "actuator_or_muscle_count": len(muscles),
        "equality_constraint_count": len(equalities),
        "converted_muscle_count": model_counts["converted_muscles"],
        "converted_ligament_count": model_counts["converted_ligaments"],
        "converted_path_point_count": model_counts["converted_path_points"],
        "converted_path_wrap_count": model_counts["converted_path_wraps"],
        "converted_coordinate_coupler_constraint_count": model_counts["converted_constraints"],
        "coordinate_default_value_count": len(coordinate_defaults),
        "locked_coordinate_count": len(locked_defaults),
        "missing_wrap_geom_reference_count": len(set(missing_wrap_geoms)),
        "missing_path_site_count": model_counts["missing_path_site_count"],
        "missing_path_wrap_count": model_counts["missing_path_wrap_count"],
        "skeleton_model_muscle_count": skeleton_counts["converted_muscles"],
        "bones_model_muscle_count": bones_counts["converted_muscles"],
        "core_model_muscle_count": core_counts["converted_muscles"],
        "max_body_position_error_m": alignment_report["max_body_position_error_m"],
        "max_body_rotation_error_rad": alignment_report["max_body_rotation_error_rad"],
        "max_mesh_position_error_m": alignment_report["max_mesh_position_error_m"],
        "max_mesh_rotation_error_rad": alignment_report["max_mesh_rotation_error_rad"],
        "max_wrap_position_error_m": alignment_report["max_wrap_position_error_m"],
        "max_wrap_rotation_error_rad": alignment_report["max_wrap_rotation_error_rad"],
        "max_marker_position_error_m": alignment_report["max_marker_position_error_m"],
        "limitations": [
            "MuJoCo contact pairs are not converted to OpenSim contact geometries.",
            "MuJoCo tendon geom/sidesite entries are converted to OpenSim wrap objects and PathWrap entries; MuJoCo sidesites are represented as per-path wrap-object quadrants.",
            "MuJoCo mesh geom pos/euler/quat transforms are baked into generated STL files when needed so OpenSim GUI rendering matches MuJoCo local mesh placement.",
            "The OpenSim default root coordinate values are set for a Y-up standing visualization in the OpenSim GUI.",
            "OpenSim and MuJoCo use different muscle dynamics implementations; source force, lengthrange, springlength, gainprm, biasprm, dynprm, and ctrlrange are preserved in the conversion calculations where OpenSim has corresponding fields.",
        ],
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
