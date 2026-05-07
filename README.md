# MimicMSK OpenSim vs MuJoCo Equivalence Tool / MimicMSK OpenSim 与 MuJoCo 等价性检查工具

This repository contains MimicMSK model assets and a Python package, `msk_equivalence`, for systematically comparing an OpenSim full-body musculoskeletal model against a MuJoCo MJCF model.

本仓库包含 MimicMSK 模型资源和 Python 包 `msk_equivalence`，用于系统性比较 OpenSim 全身肌骨模型与 MuJoCo MJCF 模型。

The tool is not a visual diff. Two models can look similar while using different joint axes, body frames, inertias, tendon paths, muscle parameters, passive forces or contact models. Those differences are exactly what can break inverse dynamics, muscle-control reinforcement learning, and video-SMPL-OpenSim-MuJoCo pipelines.

该工具不是视觉差异对比工具。两个模型外观可能很相似，但关节轴、刚体坐标系、惯量、肌腱路径、肌肉参数、被动力或接触模型可能不同。这些差异正是可能破坏逆动力学、肌肉控制强化学习，以及 video-SMPL-OpenSim-MuJoCo 流程的关键原因。

## Model Files / 模型文件

The current repository includes:

当前仓库包含：

- OpenSim model / OpenSim 模型：`MimicMSK_Model_opensim/MimicMSK_OpenSim.osim`
- MuJoCo model / MuJoCo 模型：`MimicMSK_Model_mujoco/body/myofullbody.xml`

The command line does not hard-code these paths. You may also use paths such as:

命令行不会硬编码这些路径。你也可以使用如下路径：

- `models/MimicMSK_OpenSim.osim`
- `models/mimic_msk_mujoco.xml`
- `configs/model_mapping.yaml`

If your target files do not exist, provide them yourself and pass their paths through CLI arguments.

如果你的目标文件不存在，需要自行提供，并通过 CLI 参数传入对应路径。

## Equivalence Levels / 等价性层级

- Level 0: visual/structural equivalence, including topology and naming coverage. / 第 0 级：视觉/结构等价性，包括拓扑结构和命名覆盖情况。
- Level 1: kinematic equivalence, including body, marker/site and COM positions under matched poses. / 第 1 级：运动学等价性，包括匹配姿态下的刚体、marker/site 和质心位置。
- Level 2: musculoskeletal geometry equivalence, especially muscle-tendon length and moment arm. / 第 2 级：肌骨几何等价性，尤其是肌肉-肌腱长度和力臂。
- Level 3: dynamic equivalence, especially mass, COM, inertia, inverse dynamics torque and muscle-generated torque. / 第 3 级：动力学等价性，尤其是质量、质心、惯量、逆动力学力矩和肌肉产生的力矩。
- Level 4: task behavior equivalence, including standing, running, landing, squat and badminton motion behavior. / 第 4 级：任务行为等价性，包括站立、跑步、落地、深蹲和羽毛球动作行为。

The report marks each level as `passed`, `warning`, `failed` or `not evaluated`.

报告会将每个层级标记为 `passed`、`warning`、`failed` 或 `not evaluated`。

## Install / 安装

Python 3.10+ is expected.

需要 Python 3.10 或更高版本。

Windows PowerShell:

```powershell
pip install -r requirements.txt
pip install -e .
```

Linux/macOS Bash:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

For real model-backed checks, also install:

如果要运行真实模型支持的检查，还需要安装：

- OpenSim Python API from your OpenSim 4.x installation. / 来自 OpenSim 4.x 安装目录的 OpenSim Python API。
- MuJoCo Python binding, usually `pip install mujoco`. / MuJoCo Python 绑定，通常可通过 `pip install mujoco` 安装。

The loader reports a clear error if either backend is missing.

如果缺少任一后端，加载器会报告清晰的错误信息。

## Quick Test / 快速测试

After cloning the repository, run the built-in smoke test first. It verifies the Python package, mapping parser, topology check, kinematics check and report writer without requiring OpenSim or MuJoCo runtime bindings.

克隆仓库后，建议先运行内置 smoke test。该测试无需 OpenSim 或 MuJoCo 运行时绑定，即可验证 Python 包、mapping 解析器、拓扑检查、运动学检查和报告写出功能。

Windows PowerShell:

```powershell
python -m unittest discover -s test -p "test_*.py"
```

Linux/macOS Bash:

```bash
python3 -m unittest discover -s test -p "test_*.py"
```

For model-backed comparison, install the OpenSim Python API and MuJoCo binding, then run:

对于真实模型支持的对比，请先安装 OpenSim Python API 和 MuJoCo 绑定，然后运行：

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File examples/run_compare.ps1
```

Linux/macOS Bash:

```bash
python3 -m msk_equivalence.compare \
  --osim MimicMSK_Model_opensim/MimicMSK_OpenSim.osim \
  --mjcf MimicMSK_Model_mujoco/body/myofullbody.xml \
  --mapping configs/model_mapping.yaml \
  --out results/equivalence_report
```

Reports are written under `results/equivalence_report/`, which is intentionally ignored by Git.

报告会写入 `results/equivalence_report/`，该目录会被 Git 有意忽略。

The report also writes worst-case diagnostics under `results/equivalence_report/diagnostics/`. Use these CSV files first when deciding which model geometry to change.

报告还会在 `results/equivalence_report/diagnostics/` 下写出最坏样本诊断 CSV。决定修改哪部分模型几何前，优先查看这些文件。

To make the command fail in CI when the equivalence gate is not satisfied, add `--fail-on-gate`:

如果希望门控不通过时在 CI 中返回失败状态，可以加 `--fail-on-gate`：

```bash
PYTHONPATH=src MSK_EQUIVALENCE_SKIP_PLOTS=1 python3 -m msk_equivalence.compare \
  --osim MimicMSK_Model_opensim/MimicMSK_OpenSim.osim \
  --mjcf MimicMSK_Model_mujoco/body/myofullbody.xml \
  --mapping configs/model_mapping.yaml \
  --out results/equivalence_report \
  --fail-on-gate
```

The current dynamics experiment protocol is documented in `docs/dynamics_equivalence_experiments.md`; current model findings are summarized in `docs/equivalence_findings.md`.

当前动力学实验协议见 `docs/dynamics_equivalence_experiments.md`；当前模型问题汇总见 `docs/equivalence_findings.md`。

## Repository Layout / 仓库结构

- `src/msk_equivalence/`: Python package and CLI implementation. / Python 包和 CLI 实现。
- `test/`: smoke tests and all test-only files. / smoke tests 和所有仅用于测试的文件。
- `configs/`: mapping template and generated mapping. / mapping 模板和生成后的 mapping。
- `examples/`: example commands and starter mapping. / 示例命令和初始 mapping。
- `resources/opensim_defaults/`: OpenSim default XML snippets moved out of the repository root. / 从仓库根目录移出的 OpenSim 默认 XML 片段。
- `results/`: recommended output location for generated reports. / 建议用于存放生成报告的输出目录。

## Mapping File / 映射文件

Copy the template and edit names:

复制模板并编辑名称：

Windows PowerShell:

```powershell
Copy-Item configs/model_mapping.template.yaml configs/model_mapping.yaml
```

Linux/macOS Bash:

```bash
cp configs/model_mapping.template.yaml configs/model_mapping.yaml
```

`mapping.yaml` defines:

`mapping.yaml` 定义：

- body mapping: OpenSim body name to MuJoCo body name / 刚体映射：OpenSim 刚体名称到 MuJoCo 刚体名称
- coordinate mapping: OpenSim coordinate name to MuJoCo joint or `qpos[i]` / 坐标映射：OpenSim coordinate 名称到 MuJoCo joint 或 `qpos[i]`
- marker/site mapping / marker/site 映射
- muscle mapping: OpenSim muscle to MuJoCo tendon or actuator / 肌肉映射：OpenSim muscle 到 MuJoCo tendon 或 actuator
- contact mapping / 接触映射
- pose samples and optional joint sweep values / 姿态样本和可选的关节扫描值
- coordinate sign flips when needed / 必要时的坐标符号翻转

An example starter file is in `examples/model_mapping.example.yaml`.

示例初始文件位于 `examples/model_mapping.example.yaml`。

## Run / 运行

Windows PowerShell:

```powershell
python -m msk_equivalence.compare `
  --osim MimicMSK_Model_opensim/MimicMSK_OpenSim.osim `
  --mjcf MimicMSK_Model_mujoco/body/myofullbody.xml `
  --mapping configs/model_mapping.yaml `
  --out results/equivalence_report
```

Linux/macOS Bash:

```bash
python3 -m msk_equivalence.compare \
  --osim MimicMSK_Model_opensim/MimicMSK_OpenSim.osim \
  --mjcf MimicMSK_Model_mujoco/body/myofullbody.xml \
  --mapping configs/model_mapping.yaml \
  --out results/equivalence_report
```

## Equivalence Gate / 等价性门控

For training or regression checks, run the comparison as a gate. The command exits with code `1` unless the configured verdict is `equivalent`:

训练前或回归测试时，可以把比较工具当作门控运行。如果最终 verdict 不是 `equivalent`，命令会以退出码 `1` 结束：

```bash
PYTHONPATH=src MSK_EQUIVALENCE_SKIP_PLOTS=1 python3 -m msk_equivalence.compare \
  --osim MimicMSK_Model_opensim/MimicMSK_OpenSim.osim \
  --mjcf MimicMSK_Model_mujoco/body/myofullbody.xml \
  --mapping configs/model_mapping.yaml \
  --out results/equivalence_report \
  --fail-on-gate
```

The gate is layered:

门控按层级判断：

- Level 0: topology and naming coverage. / 拓扑和命名覆盖。
- Level 1: rigid-body kinematics, marker/site positions and COM. / 刚体运动学、marker/site 和质心。
- Level 2: muscle-tendon length and direct independent-coordinate moment arms. / 肌肉-肌腱长度和独立坐标力臂。
- Level 3: mass, COM, inertia and dynamics adapters when configured. / 质量、质心、惯性以及已配置的动力学适配器。
- Level 4: task-level contact and forward behavior when adapters are implemented. / 接触和正向任务行为。

Dependent coordinates from OpenSim `CoordinateCouplerConstraint` are reported in `moment_arm_error.csv`, but they are not used as direct moment-arm pass/fail gates because they require constraint-chain-aware validation.

OpenSim `CoordinateCouplerConstraint` 的依赖坐标会记录在 `moment_arm_error.csv` 中，但不会作为直接力臂通过/失败门控，因为它们需要基于约束链式关系的验证。

You can run a subset of checks:

你可以只运行部分检查：

Windows PowerShell:

```powershell
python -m msk_equivalence.compare --osim path/to/model.osim --mjcf path/to/model.xml --mapping configs/model_mapping.yaml --out results/equivalence_report --checks topology inertial kinematics muscle_length moment_arm
```

Linux/macOS Bash:

```bash
python3 -m msk_equivalence.compare --osim path/to/model.osim --mjcf path/to/model.xml --mapping configs/model_mapping.yaml --out results/equivalence_report --checks topology inertial kinematics muscle_length moment_arm
```

## Outputs / 输出

The output directory contains:

输出目录包含：

- `index.md`: human-readable summary / 人类可读的摘要
- `summary.json`: machine-readable status and metrics / 机器可读的状态和指标
- `topology_summary.csv`, `topology_mismatch.json`
- `convention_report.md`
- `inertial_body_comparison.csv`, `total_mass_comparison.json`
- `kinematics_body_pose_error.csv`, `kinematics_marker_site_error.csv`, `whole_body_com_error.csv`
- `joint_sweep_errors.csv`, `possible_sign_flip_warnings.json`
- `muscle_length_error.csv`, `plots/muscle_length/*.png`
- `moment_arm_error.csv`, `plots/moment_arm/*.png`
- placeholder outputs for contact, inverse dynamics, forward dynamics, passive forces and muscle torque when adapters are not yet implemented / 当适配器尚未实现时，会生成接触、逆动力学、正向动力学、被动力和肌肉力矩相关的占位输出。

## What Is Automatic / 自动完成的内容

The MVP automatically loads OpenSim and MuJoCo models, reads mapping YAML, compares topology, inertial values, body/marker/site kinematics, muscle-tendon length, and moment arms using MuJoCo finite differences when necessary.

该 MVP 会自动加载 OpenSim 和 MuJoCo 模型，读取 mapping YAML，并比较拓扑、惯性参数、刚体/marker/site 运动学、肌肉-肌腱长度，以及必要时通过 MuJoCo 有限差分计算的力臂。

## What Needs Manual Validation / 需要人工验证的内容

Manual review is required for model conventions, frame alignment, pelvis/root definitions, OpenSim units from provenance, contact model tuning, muscle model functional equivalence, passive force decomposition, and long-horizon task behavior. Inertia tensor comparisons are only meaningful when the compared frames are aligned.

模型约定、坐标系对齐、骨盆/root 定义、来自来源信息的 OpenSim 单位、接触模型调参、肌肉模型功能等价性、被动力分解，以及长时域任务行为都需要人工检查。只有在被比较的坐标系已经对齐时，惯性张量比较才有意义。

## Why This Matters / 为什么重要

For muscle RL and video-to-SMPL-to-OpenSim-to-MuJoCo workflows, visual similarity is not enough. Policies and inverse dynamics depend on whether coordinates, moment arms, mass distribution, contact behavior and muscle force generation are consistent. This tool separates those layers so you can identify whether failures come from retargeting, kinematics, muscle geometry, dynamics or task-level simulation behavior.

对于肌肉强化学习和 video-to-SMPL-to-OpenSim-to-MuJoCo 工作流来说，仅有视觉相似是不够的。策略和逆动力学依赖于坐标、力臂、质量分布、接触行为和肌肉力生成是否一致。该工具将这些层级拆开，帮助你判断失败究竟来自重定向、运动学、肌肉几何、动力学，还是任务级仿真行为。
