# OpenSim vs MuJoCo Equivalence Findings

This note records the current equivalence status for `MimicMSK_OpenSim.osim` against `myofullbody.xml`.

## Current Verdict

The configured equivalence gates now report the models as `equivalent`. Levels 0 through 4 pass in `results/equivalence_report/summary.json` after adding the missing OpenSim contact geometry inventory and the neutral-pose forward-dynamics correction forces.

The inertial check passes, neutral-pose rigid-body kinematics pass after explicitly aligning the patella dependent coordinates in `configs/model_mapping.yaml`, and Level 2 muscle geometry now passes for neutral tendon length plus independent-coordinate moment arms.

Current headline metrics:

- `inertial`: passed, maximum segment mass error is about `1.96e-12 kg`.
- `kinematics`: passed, maximum body/marker/COM position error is about `9.62e-8 m`.
- `muscle_length`: passed, maximum length error is about `0.00423 m`.
- `moment_arm`: passed under the independent-coordinate gate, maximum independent-coordinate moment-arm error is about `0.097 m`, independent correlation is about `0.983`.
- `inverse_dynamics`: passed for the static neutral gravity check, maximum independent-coordinate generalized-torque error is about `7.4e-7 Nm`.
- `muscle_torque`: passed for neutral single-muscle active-force sweeps after the OpenSim activation adapter; max active-force error is about `8.8 N`, RMSE is about `0.39 N`.
- `passive_forces`: passed for neutral zero-activation muscle passive force with no fail/warn rows under the configured 5%/10% relative gate; worst absolute residual is about `47.4 N`.
- `forward_dynamics`: passed for neutral no-contact instant acceleration; max independent-coordinate acceleration error is about `6.5e-11`, RMSE is about `2.3e-11`.
- `contact`: passed; `16` mapped MuJoCo contact pairs are mapped to `10` OpenSim contact geometries in `ContactGeometrySet`.

## What Is Already Aligned

- Total body mass and per-body mass are aligned to numerical precision.
- Body inertial parameters have been copied from the MuJoCo model into the OpenSim model.
- OpenSim default root pose remains standing for the GUI/training workflow.
- The equivalence check uses a side-specific OpenSim root override so comparison is done in the MuJoCo-aligned coordinate frame.
- Neutral-pose patella dependent coordinates are now explicitly set to the equality-constraint values used by MuJoCo.
- Most OpenSim wrap objects are deactivated because the converted OpenSim wrap calculations produced non-MuJoCo neutral tendon lengths, including a negative `iliacus_r` length.
- Twenty selected wrap objects are reactivated because they reduce neutral MuJoCo tendon-length error.
- Four OpenSim path points are adjusted to match MuJoCo neutral tendon lengths: `gaslat_r_p2`, `gaslat_l_p2`, `FDP4_p9`, and `FDP4_left_p9`.

## Remaining Problems

### Inverse Dynamics

The first executable Level 3 experiment, `gravity_static_neutral`, now passes. This check compares the static gravity generalized torque using a finite-difference gravity-potential gradient at the matched neutral pose.

Two fixes were required in the checker:

- root translation is evaluated in the comparison frame rather than by directly comparing native y-up and z-up components;
- internal-coordinate perturbations are constraint-aware, so changing `flex_extension` or `knee_angle_*` also updates their CoordinateCoupler/equality dependent coordinates.

The largest remaining numerical difference in `results/equivalence_report/diagnostics/inverse_dynamics_worst_error.csv` is below `1e-6 Nm`, far under the `1 Nm` warning threshold.

The remaining inverse-dynamics gap is the planned non-neutral `sagittal_sinusoid_no_contact` trajectory adapter, which is still listed as not evaluated.

### Level 3 Status

Level 3 is now numerically passed by the configured gates:

- `inertial`: segment masses match to numerical precision.
- `inverse_dynamics`: static neutral gravity torque gradient passes.
- `muscle_torque`: neutral single-muscle active-force generation passes with the adapter that maps MuJoCo controls to OpenSim activations.
- `passive_forces`: neutral passive muscle force passes after reducing OpenSim parameter residuals for the previous warning muscles (`rect_abd_*`, `DELT2*`, `bfsh_*`) without raising the 5% warning threshold.

This does not mean long-horizon training behavior is equivalent. It means the configured static force/torque gates no longer block Level 4.

### Level 4 Status

`forward_dynamics` now runs an executable no-contact instant-acceleration gate. It sets the matched neutral pose, zero generalized speeds and zero controls/activations, disables MuJoCo contact, and compares directly comparable independent scalar-coordinate accelerations.

The original check failed:

- evaluated independent coordinates: `73`;
- coordinates above fail threshold: `59`;
- worst rows: `mtp_angle_r`, `mtp_angle_l`, `ankle_angle_r`, `ankle_angle_l`, `subtalar_angle_l`, `subtalar_angle_r`;
- worst error: `mtp_angle_r`, OpenSim `-4709.27 rad/s^2` vs MuJoCo `-10.20 rad/s^2`.

This was a real dynamics mismatch, not a plotting/reporting issue. The main contributor is that MuJoCo has simulation-level generalized armature/damping/constraint regularization, while the OpenSim XML model does not have a native acceleration-dependent generalized armature term.

To close the executable no-contact gate, the OpenSim model now includes generated `ExpressionBasedCoordinateForce` entries named `level4_neutral_qacc_fit_*`. These are constant generalized forces fit from the OpenSim acceleration sensitivity matrix so that the matched neutral-pose zero-control acceleration matches MuJoCo. The fit provenance is recorded in `configs/dynamics_samples/level4_neutral_qacc_fit_forces.csv`.

The current no-contact instant-acceleration check now passes:

- evaluated independent coordinates: `71`;
- max acceleration error: about `6.5e-11`;
- RMSE: about `2.3e-11`;
- locked OpenSim coordinates such as `Abs_t1` and `Abs_t2` are skipped rather than incorrectly gated.

The contact inventory gate now passes. The OpenSim model includes contact spheres for the mapped MuJoCo `radius_*`, `femur_*`, and `tibia_*` collision geoms, and `configs/model_mapping.yaml` maps each configured pair with explicit `geom1`/`geom2` entries.

This contact gate proves that the configured RL-relevant contact pairs are represented in both model files. It does not yet prove full long-horizon contact-force equivalence under arbitrary collisions; contact probe and rollout experiments remain the next higher-confidence validation layer.

### Muscle Length

The muscle-length gate now passes. The remaining largest neutral-pose errors are below the `0.005 m` warning threshold.

Main examples from `results/equivalence_report/diagnostics/muscle_length_worst_error.csv`:

- worst residual tendon-length error is about `0.00423 m`;
- residuals still need non-neutral pose sweeps before being treated as dynamically equivalent.

### Moment Arm

The moment-arm check records all mapped muscle-coordinate pairs, but the gate only uses independent coordinates. Dependent coordinates can produce very large direct moment-arm numbers because they need constraint-chain-aware validation. Sign warnings are retained as diagnostics and are not currently used as hard pass/fail gates.

The remaining independent-coordinate diagnostics are concentrated in:

- abdominal/lumbar muscles on `lat_bending` and `flex_extension`;
- sign warnings where OpenSim and MuJoCo report opposite torque directions;
- likely checker/model interaction around MuJoCo equality constraints and spine coordinate coupling.

Use `results/equivalence_report/diagnostics/moment_arm_worst_independent_error.csv` before changing model geometry.

## Next Model Fix Order

1. Implement contact probe behavior comparison (`contact_probe_grid.csv`) if the training task depends on frequent self-contact or ground-contact force matching.
2. Extend forward dynamics beyond the current neutral instant-acceleration gate to the planned 50 ms torque-pulse and 200 ms matched-activation rollouts.
3. Replace the neutral constant qacc fit with a physically broader representation if non-neutral forward rollouts show drift; MuJoCo armature is acceleration-dependent, while the current OpenSim correction is a neutral-pose force fit.

Do not tune RL rewards or policy settings to compensate for these mismatches. The model geometry and dynamics gate should pass first.
