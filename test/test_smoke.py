from __future__ import annotations

import math
import shutil
import sys
import unittest
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from msk_equivalence.checks import kinematics, topology
from msk_equivalence.mapping import MappingConfig
from msk_equivalence.report.generate_report import generate


class FakeModel:
    path = Path("fake")
    bodies = ["pelvis", "femur"]
    coordinates = ["hip"]
    muscles = ["iliopsoas"]
    actuators = ["iliopsoas"]
    markers = ["asis"]
    joints = ["root"]

    def hierarchy(self):
        return [{"joint": "hip", "parent": "pelvis", "child": "femur"}]

    def set_pose(self, values):
        self.values = values

    def body_position(self, name):
        return [0.0, 0.0, 0.0] if name == "pelvis" else [1.0, 0.0, 0.0]

    def marker_position(self, name):
        return [0.0, 1.0, 0.0]

    def whole_body_com(self):
        return [0.5, 0.0, 0.0]


class SmokeTests(unittest.TestCase):
    def test_mapping_topology_kinematics_and_report(self):
        root = Path.cwd() / "test" / f"_smoke_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        try:
            mapping_path = root / "mapping.yaml"
            mapping_path.write_text(
                """
bodies:
  - opensim: pelvis
    mujoco: pelvis
  - opensim: femur
    mujoco: femur
coordinates:
  - opensim: hip
    mujoco: hip
markers:
  - opensim: asis
    mujoco: asis
muscles:
  - opensim: iliopsoas
    mujoco: iliopsoas
pose_samples:
  - name: neutral
    q: {}
""",
                encoding="utf-8",
            )
            mapping = MappingConfig.load(mapping_path)
            out = root / "out"
            model_a = FakeModel()
            model_b = FakeModel()

            topo = topology.run(model_a, model_b, mapping, out)
            kin = kinematics.run(model_a, model_b, mapping, out)
            summary = generate(out, {"topology": topo, "kinematics": kin}, {"osim": "a.osim", "mjcf": "b.xml", "mapping": str(mapping_path)})

            self.assertEqual(topo["status"], "passed")
            self.assertEqual(kin["status"], "passed")
            self.assertTrue(math.isclose(kin["max_error"], 0.0))
            self.assertTrue((out / "index.md").exists())
            self.assertTrue((out / "summary.json").exists())
            self.assertEqual(summary["inputs"]["osim"], "a.osim")
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
