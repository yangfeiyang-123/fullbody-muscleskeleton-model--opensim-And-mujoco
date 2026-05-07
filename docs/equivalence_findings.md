# OpenSim vs MuJoCo Equivalence Findings

This note records the current equivalence status for `MimicMSK_OpenSim.osim` against `myofullbody.xml`.

## Current Verdict

The models are not yet dynamically equivalent. The current gate result is `partially equivalent`.

The inertial check passes, and the neutral-pose rigid-body kinematics now pass after explicitly aligning the patella dependent coordinates in `configs/model_mapping.yaml`.

Current headline metrics:

- `inertial`: passed, maximum segment mass error is about `1.96e-12 kg`.
- `kinematics`: passed, maximum body/marker/COM position error is about `9.62e-8 m`.
- `muscle_length`: warning, maximum length error is about `0.020 m`.
- `moment_arm`: warning, maximum independent-coordinate moment-arm error is about `0.097 m`.

## What Is Already Aligned

- Total body mass and per-body mass are aligned to numerical precision.
- Body inertial parameters have been copied from the MuJoCo model into the OpenSim model.
- OpenSim default root pose remains standing for the GUI/training workflow.
- The equivalence check uses a side-specific OpenSim root override so comparison is done in the MuJoCo-aligned coordinate frame.
- Neutral-pose patella dependent coordinates are now explicitly set to the equality-constraint values used by MuJoCo.
- OpenSim wrap objects are deactivated because the converted OpenSim wrap calculations produced non-MuJoCo neutral tendon lengths, including a negative `iliacus_r` length.

## Remaining Problems

### Muscle Length

The muscle-length gate no longer fails, but it still warns. The largest current errors are concentrated in a small number of tendon paths.

Main examples from `results/equivalence_report/diagnostics/muscle_length_worst_error.csv`:

- `gasmed_l`, `gasmed_r`: largest residual length error is about `0.020 m`.
- `TMAJ`, `LAT*`, `SUP`, `gaslat_*`, `FDP4*`: residual errors are below the failure threshold but above the warning threshold.
- These residuals need non-neutral pose sweeps before being treated as dynamically equivalent.

### Moment Arm

The moment-arm check records all mapped muscle-coordinate pairs, but the gate only uses independent coordinates. Dependent coordinates can produce very large direct moment-arm numbers because they need constraint-chain-aware validation.

The independent-coordinate warnings are concentrated in:

- abdominal/lumbar muscles on `lat_bending` and `flex_extension`;
- sign warnings where OpenSim and MuJoCo report opposite torque directions;
- likely checker/model interaction around MuJoCo equality constraints and spine coordinate coupling.

Use `results/equivalence_report/diagnostics/moment_arm_worst_independent_error.csv` before changing model geometry.

## Next Model Fix Order

1. Add non-neutral pose samples for muscle length and moment-arm checks so neutral-pose agreement is not over-interpreted.
2. Fix or replace the MuJoCo/OpenSim moment-arm adapter for equality-constrained coordinates before tuning abdominal/lumbar muscle geometry.
3. Implement Level 3 inverse dynamics, muscle torque and passive force adapters.
4. Only after Level 3 passes, use short forward dynamics rollouts as a behavior gate.

Do not tune RL rewards or policy settings to compensate for these mismatches. The model geometry and dynamics gate should pass first.
