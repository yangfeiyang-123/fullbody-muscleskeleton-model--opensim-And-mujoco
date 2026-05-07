# OpenSim vs MuJoCo Equivalence Findings

This note records the current equivalence status for `MimicMSK_OpenSim.osim` against `myofullbody.xml`.

## Current Verdict

The models are not equivalent for RL/dynamics interchange yet. Level 3 now passes, but Level 4 fails because the first executable forward-dynamics smoke test shows large acceleration mismatches at the matched neutral pose.

The inertial check passes, neutral-pose rigid-body kinematics pass after explicitly aligning the patella dependent coordinates in `configs/model_mapping.yaml`, and Level 2 muscle geometry now passes for neutral tendon length plus independent-coordinate moment arms.

Current headline metrics:

- `inertial`: passed, maximum segment mass error is about `1.96e-12 kg`.
- `kinematics`: passed, maximum body/marker/COM position error is about `9.62e-8 m`.
- `muscle_length`: passed, maximum length error is about `0.00423 m`.
- `moment_arm`: passed under the independent-coordinate gate, maximum independent-coordinate moment-arm error is about `0.097 m`, independent correlation is about `0.983`.
- `inverse_dynamics`: passed for the static neutral gravity check, maximum independent-coordinate generalized-torque error is about `7.4e-7 Nm`.
- `muscle_torque`: passed for neutral single-muscle active-force sweeps after the OpenSim activation adapter; max active-force error is about `8.8 N`, RMSE is about `0.39 N`.
- `passive_forces`: passed for neutral zero-activation muscle passive force with no fail/warn rows under the configured 5%/10% relative gate; worst absolute residual is about `47.4 N`.
- `forward_dynamics`: failed for neutral no-contact instant acceleration; max independent-coordinate acceleration error is about `4699 rad/s^2`, RMSE is about `905 rad/s^2`.

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

### Level 4 Blocking Failure

`forward_dynamics` now runs an executable no-contact instant-acceleration gate. It sets the matched neutral pose, zero generalized speeds and zero controls/activations, disables MuJoCo contact, and compares directly comparable independent scalar-coordinate accelerations.

The check fails:

- evaluated independent coordinates: `73`;
- coordinates above fail threshold: `59`;
- worst rows: `mtp_angle_r`, `mtp_angle_l`, `ankle_angle_r`, `ankle_angle_l`, `subtalar_angle_l`, `subtalar_angle_r`;
- worst error: `mtp_angle_r`, OpenSim `-4709.27 rad/s^2` vs MuJoCo `-10.20 rad/s^2`.

This is a real dynamics mismatch, not a plotting/reporting issue. Likely contributors are MuJoCo joint armature/damping/constraint regularization and backend-specific coordinate dynamics that are not represented by the current OpenSim model. For example, MuJoCo has nonzero armature and damping on the worst lower-limb coordinates, while the OpenSim model has no equivalent generalized armature term.

`contact` remains a warning. The OpenSim model has an empty `ContactGeometrySet`, while the MuJoCo model has configured collision/contact pairs; contact behavior still cannot be claimed equivalent.

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

1. Fix the Level 4 acceleration mismatch before claiming the models can be used interchangeably for training.
2. Decide how to represent MuJoCo generalized armature/damping/regularization in OpenSim, or explicitly remove/disable those MuJoCo terms in the training reference. Since the current goal is to fix OpenSim toward MuJoCo, the OpenSim-side equivalent must be modeled rather than ignored.
3. Add a generalized passive-torque gate, because neutral passive muscle force passing is weaker than matching passive generalized accelerations.
4. Add OpenSim contact geometry/forces or explicitly remove MuJoCo contacts from the equivalence target; the current OpenSim `ContactGeometrySet` is empty.
5. After instant acceleration passes, run the planned 50 ms torque-pulse, 200 ms matched-activation, and contact-drop rollouts.

Do not tune RL rewards or policy settings to compensate for these mismatches. The model geometry and dynamics gate should pass first.
