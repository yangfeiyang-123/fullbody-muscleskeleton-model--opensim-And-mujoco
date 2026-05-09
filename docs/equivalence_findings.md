# OpenSim vs MuJoCo Equivalence Findings

This note records the current equivalence status for `MimicMSK_OpenSim.osim` against `myofullbody.xml`.

## Current Verdict

Latest audit status: see `docs/completion_audit.md` and `docs/level4_failure_localization.md` for the current evidence trail.

As of the latest checked artifacts:

- Level 0-3 configured gates remain passed on the current repository model.
- Level 3.5 generalized torque decomposition is passed on sparse random activation vectors at the tested neutral pose: 324 evaluated rows, max torque error `0.0983299002645679 Nm`, RMSE `0.009914946676079605 Nm`, missing coverage rows `0`.
- Strict native Level 4 remains failed. The current OpenSim `.osim` cannot be reported as a behaviorally equivalent or RL-interchangeable MuJoCo counterpart.
- Diagnostic adapter probes are useful for localization but are not native `.osim` Level 4 passes.

Do not treat older report paths in this file as the latest state unless they match the audit documents above.

The strict equivalence gates do not currently support calling `MimicMSK_OpenSim.osim` a validated MuJoCo counterpart. Earlier configured gates passed, but the stricter Level 3.5 generalized-torque decomposition and Level 4 random-activation / rollout gates expose remaining dynamic mismatches.

Current strict status from the latest audit:

- Level 0 through Level 3: passed under the configured gates used in `/tmp/equiv_0_35_after_wrapfix/summary.json`; these must be re-run after every accepted model repair.
- Level 3.5 generalized torque: passed in `/tmp/equiv_0_35_after_wrapfix/summary.json`.
- Level 4 random activation qacc: failed in `/tmp/equiv_l4_after_wrapfix/summary.json`.
- Level 4 no-contact short rollout: failed in `/tmp/equiv_l4_after_wrapfix/summary.json`.
- RL action/state distribution rollout: not yet passed.
- Contact rollout: not yet behaviorally validated; only contact inventory/mapping is available.

Therefore the correct current verdict is `not equivalent` for RL-interchangeable behavior.

The inertial check passes, neutral-pose rigid-body kinematics pass after explicitly aligning the patella dependent coordinates in `configs/model_mapping.yaml`, and Level 2 muscle geometry now passes for neutral tendon length plus independent-coordinate moment arms.

Current headline metrics:

- `inertial`: passed, maximum segment mass error is about `1.96e-12 kg`.
- `kinematics`: passed, maximum body/marker/COM position error is about `9.62e-8 m`.
- `muscle_length`: passed, maximum length error is about `0.00423 m`.
- `moment_arm`: passed under the independent-coordinate gate, maximum independent-coordinate moment-arm error is about `0.097 m`, independent correlation is about `0.983`.
- `inverse_dynamics`: passed for the static neutral gravity check, maximum independent-coordinate generalized-torque error is about `7.4e-7 Nm`.
- `muscle_torque`: passed for neutral single-muscle active-force sweeps after the OpenSim activation adapter. With the denser activation adapter grid, the targeted run in `results/equivalence_report_level35/summary.json` reports max active-force error about `2.18 N`.
- `passive_forces`: passed for neutral zero-activation muscle passive force under the configured gate; latest audited max passive-force residual is about `8.30 N`, RMSE about `0.997 N`.
- `forward_dynamics`: failed under strict native Level 4. Neutral no-contact instant acceleration still has max qacc error about `125.57`; random activation and rollout fail by much larger margins.
- `contact`: passed; `16` mapped MuJoCo contact pairs are mapped to `10` OpenSim contact geometries in `ContactGeometrySet`.
- `generalized_torque`: passed under the current sparse random activation Level 3.5 gate. The current gate uses finite-difference total moment arms along the coupled coordinate path for directly comparable independent coordinates. A direct dependent-coordinate projection was tested and rejected because OpenSim `computeMomentArm` for constrained dependent coordinates is not a plain partial derivative. Latest audited max active generalized-torque error is about `0.0983 Nm`, RMSE about `0.00991 Nm`, with missing coverage rows `0`.
- `random_activation_qacc`: failed. Worst current diagnostics still include `mtp_angle_r`, `mtp_angle_l`, `ankle_angle_r`, `md3_flexion_r`, `mp_flexion_r/l`, and other small-inertia distal coordinates.

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

This contact gate proves only that the configured RL-relevant contact pairs are represented in both model files. It does not prove contact-force, contact-timing, impulse, or foot-slip equivalence.

The strict Level 4 random-activation qacc gate now fails. The largest errors are not explained solely by active muscle force mismatch. The diagnostics show two important causes:

- remaining active generalized-torque mismatches after finite-difference coupled-coordinate moment-arm comparison;
- very large qacc amplification on small-inertia distal coordinates, often when no mapped active muscle directly contributes to that coordinate.

This second class is consistent with MuJoCo model semantics that include joint-space armature/regularization on many distal joints, while the OpenSim XML model has no native acceleration-dependent generalized armature term. The new `diagnostics/mujoco_coordinate_dynamics_audit.csv` confirms that the worst distal rows have large armature fractions of the MuJoCo mass diagonal, for example `mtp_angle_r/l` are about `98.8%` armature. The in-memory diagnostic with MuJoCo armature zeroed reduces the random-activation qacc max error from about `48623` to about `17611`, so armature is a major source but not the only source. Constant `ExpressionBasedCoordinateForce` entries can match a neutral zero-control acceleration sample, but they do not reproduce arbitrary activation-dependent acceleration response.

### Muscle Length

The muscle-length gate now passes. The remaining largest neutral-pose errors are below the `0.005 m` warning threshold.

Main examples from `results/equivalence_report/diagnostics/muscle_length_worst_error.csv`:

- worst residual tendon-length error is about `0.00423 m`;
- residuals still need non-neutral pose sweeps before being treated as dynamically equivalent.

### Moment Arm

The moment-arm check records all mapped muscle-coordinate pairs, but the Level 2 gate only uses independent coordinates. Dependent coordinates remain diagnostic-only because OpenSim `computeMomentArm` for constrained dependent coordinates does not behave like a plain finite-difference partial derivative. The Level 3.5 gate therefore compares coupled finite-difference total moment arms for independent coordinates instead of projecting direct dependent-coordinate moment arms.

The remaining independent-coordinate diagnostics are concentrated in:

- abdominal/lumbar muscles on `lat_bending` and `flex_extension`;
- sign warnings where OpenSim and MuJoCo report opposite torque directions;
- likely checker/model interaction around MuJoCo equality constraints and spine coordinate coupling.

Use `results/equivalence_report/diagnostics/moment_arm_worst_independent_error.csv` before changing model geometry.

## Next Model Fix Order

1. Continue reducing Level 3.5 residuals without relaxing thresholds. The next targets are `hip_rotation_l` (`addmagProx_l` / `addmagProx_r` path moment arms) and `elbow_flex_r` (`ECRL` sign/moment-arm mismatch).
2. Re-run Level 0-3 after any OpenSim path/wrap change. Do not accept a Level 3.5 improvement if independent-coordinate muscle length/moment-arm, kinematics, or inertial checks regress.
3. Keep the MuJoCo armature audit in Level 4. If full-body strict equivalence is required, OpenSim needs an explicit, validated surrogate for MuJoCo armature/regularization; neutral constant qacc-fit forces are not enough.
4. Only after Level 3.5 passes should random-activation qacc and staged 10/50/100/200 ms no-contact rollouts be used as repair targets.
5. Do not claim arbitrary long-horizon bitwise identity. The report must state the tested pose/action distributions, contact limitations, integrator differences, and known model-semantics gaps.

Do not tune RL rewards or policy settings to compensate for these mismatches. The model geometry and dynamics gate should pass first.
