# OpenSim vs MuJoCo Equivalence Findings

This note records the current equivalence status for `MimicMSK_OpenSim.osim` against `myofullbody.xml`.

## Current Verdict

The models are not yet dynamically equivalent. The current gate result is `not equivalent`.

The inertial check passes, and the neutral-pose rigid-body kinematics now pass after explicitly aligning the patella dependent coordinates in `configs/model_mapping.yaml`.

Current headline metrics:

- `inertial`: passed, maximum segment mass error is about `1.96e-12 kg`.
- `kinematics`: passed, maximum body/marker/COM position error is about `9.62e-8 m`.
- `muscle_length`: failed, maximum length error is about `0.229 m`.
- `moment_arm`: failed, maximum independent-coordinate moment-arm error is about `0.104 m`.

## What Is Already Aligned

- Total body mass and per-body mass are aligned to numerical precision.
- Body inertial parameters have been copied from the MuJoCo model into the OpenSim model.
- OpenSim default root pose remains standing for the GUI/training workflow.
- The equivalence check uses a side-specific OpenSim root override so comparison is done in the MuJoCo-aligned coordinate frame.
- Neutral-pose patella dependent coordinates are now explicitly set to the equality-constraint values used by MuJoCo.

## Remaining Problems

### Muscle Length

The muscle-length gate still fails. The largest current errors are not global scale errors; they are concentrated in a small number of tendon paths.

Main examples from `results/equivalence_report/diagnostics/muscle_length_worst_error.csv`:

- `iliacus_r`: OpenSim length is negative, which indicates a path-wrap or path geometry problem, not a normal biological length mismatch.
- `glmax3_r`, `glmax3_l`: large mismatch around the pelvis wrap.
- `vasmed_*`, `vaslat_*`: mismatch remains after the patella pose fix, so path routing or wrap/contact approximation still differs.
- `FDP*_left`, `EPB`: hand muscle paths also need inspection before claiming full-body muscle geometry equivalence.

### Moment Arm

The moment-arm check records all mapped muscle-coordinate pairs, but the gate only uses independent coordinates. Dependent coordinates can produce very large direct moment-arm numbers because they need constraint-chain-aware validation.

The independent-coordinate failures are concentrated in:

- abdominal/lumbar muscles on `lat_bending` and `flex_extension`;
- `glmax3_*` and `iliacus_r` on hip flexion;
- sign warnings where OpenSim and MuJoCo report opposite torque directions.

Use `results/equivalence_report/diagnostics/moment_arm_worst_independent_error.csv` before changing model geometry.

## Next Model Fix Order

1. Fix `iliacus_r` first because a negative OpenSim muscle length is invalid.
2. Re-check the `IL_at_brim_r` wrap sphere quadrant/range/frame conversion. A temporary diagnostic run with that OpenSim wrap disabled removes the negative length, but it is not the final fix because the MuJoCo tendon still uses the wrap geom.
3. Re-check gluteus maximus wrap objects (`Gmax3_at_pelvis_*`) and path-wrap ranges.
4. Re-check quadriceps routing around patella after the neutral dependent-coordinate alignment.
5. Re-check abdominal muscle frame conventions and coordinate signs for `flex_extension` and `lat_bending`.
6. Only after Level 2 passes, implement Level 3 dynamics adapters.

Do not tune RL rewards or policy settings to compensate for these mismatches. The model geometry and dynamics gate should pass first.
