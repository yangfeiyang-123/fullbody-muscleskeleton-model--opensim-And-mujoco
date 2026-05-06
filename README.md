# MimicMSK OpenSim vs MuJoCo Equivalence Tool

This repository contains MimicMSK model assets and a Python package, `msk_equivalence`, for systematically comparing an OpenSim full-body musculoskeletal model against a MuJoCo MJCF model.

The tool is not a visual diff. Two models can look similar while using different joint axes, body frames, inertias, tendon paths, muscle parameters, passive forces or contact models. Those differences are exactly what can break inverse dynamics, muscle-control reinforcement learning, and video-SMPL-OpenSim-MuJoCo pipelines.

## Model Files

The current repository includes:

- OpenSim model: `MimicMSK_Model_opensim/MimicMSK_OpenSim.osim`
- MuJoCo model: `MimicMSK_Model_mujoco/body/myofullbody.xml`

The command line does not hard-code these paths. You may also use paths such as:

- `models/MimicMSK_OpenSim.osim`
- `models/mimic_msk_mujoco.xml`
- `configs/model_mapping.yaml`

If your target files do not exist, provide them yourself and pass their paths through CLI arguments.

## Equivalence Levels

- Level 0: visual/structural equivalence, including topology and naming coverage.
- Level 1: kinematic equivalence, including body, marker/site and COM positions under matched poses.
- Level 2: musculoskeletal geometry equivalence, especially muscle-tendon length and moment arm.
- Level 3: dynamic equivalence, especially mass, COM, inertia, inverse dynamics torque and muscle-generated torque.
- Level 4: task behavior equivalence, including standing, running, landing, squat and badminton motion behavior.

The report marks each level as `passed`, `warning`, `failed` or `not evaluated`.

## Install

Python 3.10+ is expected.

```powershell
pip install -r requirements.txt
pip install -e .
```

For real model-backed checks, also install:

- OpenSim Python API from your OpenSim 4.x installation.
- MuJoCo Python binding, usually `pip install mujoco`.

The loader reports a clear error if either backend is missing.

## Quick Test

After cloning the repository, run the built-in smoke test first. It verifies the Python package, mapping parser, topology check, kinematics check and report writer without requiring OpenSim or MuJoCo runtime bindings.

```powershell
python -m unittest discover -s test -p "test_*.py"
```

For model-backed comparison, install the OpenSim Python API and MuJoCo binding, then run:

```powershell
powershell -ExecutionPolicy Bypass -File examples/run_compare.ps1
```

Reports are written under `results/equivalence_report/`, which is intentionally ignored by Git.

## Repository Layout

- `src/msk_equivalence/`: Python package and CLI implementation.
- `test/`: smoke tests and all test-only files.
- `configs/`: mapping template and generated mapping.
- `examples/`: example commands and starter mapping.
- `resources/opensim_defaults/`: OpenSim default XML snippets moved out of the repository root.
- `results/`: recommended output location for generated reports.

## Mapping File

Copy the template and edit names:

```powershell
Copy-Item configs/model_mapping.template.yaml configs/model_mapping.yaml
```

`mapping.yaml` defines:

- body mapping: OpenSim body name to MuJoCo body name
- coordinate mapping: OpenSim coordinate name to MuJoCo joint or `qpos[i]`
- marker/site mapping
- muscle mapping: OpenSim muscle to MuJoCo tendon or actuator
- contact mapping
- pose samples and optional joint sweep values
- coordinate sign flips when needed

An example starter file is in `examples/model_mapping.example.yaml`.

## Run

```powershell
python -m msk_equivalence.compare `
  --osim MimicMSK_Model_opensim/MimicMSK_OpenSim.osim `
  --mjcf MimicMSK_Model_mujoco/body/myofullbody.xml `
  --mapping configs/model_mapping.yaml `
  --out results/equivalence_report
```

You can run a subset of checks:

```powershell
python -m msk_equivalence.compare --osim path/to/model.osim --mjcf path/to/model.xml --mapping configs/model_mapping.yaml --out results/equivalence_report --checks topology inertial kinematics muscle_length moment_arm
```

## Outputs

The output directory contains:

- `index.md`: human-readable summary
- `summary.json`: machine-readable status and metrics
- `topology_summary.csv`, `topology_mismatch.json`
- `convention_report.md`
- `inertial_body_comparison.csv`, `total_mass_comparison.json`
- `kinematics_body_pose_error.csv`, `kinematics_marker_site_error.csv`, `whole_body_com_error.csv`
- `joint_sweep_errors.csv`, `possible_sign_flip_warnings.json`
- `muscle_length_error.csv`, `plots/muscle_length/*.png`
- `moment_arm_error.csv`, `plots/moment_arm/*.png`
- placeholder outputs for contact, inverse dynamics, forward dynamics, passive forces and muscle torque when adapters are not yet implemented

## What Is Automatic

The MVP automatically loads OpenSim and MuJoCo models, reads mapping YAML, compares topology, inertial values, body/marker/site kinematics, muscle-tendon length, and moment arms using MuJoCo finite differences when necessary.

## What Needs Manual Validation

Manual review is required for model conventions, frame alignment, pelvis/root definitions, OpenSim units from provenance, contact model tuning, muscle model functional equivalence, passive force decomposition, and long-horizon task behavior. Inertia tensor comparisons are only meaningful when the compared frames are aligned.

## Why This Matters

For muscle RL and video-to-SMPL-to-OpenSim-to-MuJoCo workflows, visual similarity is not enough. Policies and inverse dynamics depend on whether coordinates, moment arms, mass distribution, contact behavior and muscle force generation are consistent. This tool separates those layers so you can identify whether failures come from retargeting, kinematics, muscle geometry, dynamics or task-level simulation behavior.
