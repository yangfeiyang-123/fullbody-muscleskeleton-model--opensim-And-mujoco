# Level 3 and Level 4 Dynamics Equivalence Experiments

This protocol defines the experiments required before claiming that `MimicMSK_OpenSim.osim` is dynamically equivalent to the MuJoCo `myofullbody.xml` model for RL training.

The current repository passes the earlier configured Level 1, Level 2, and Level 3 gates, but the stricter Level 3.5 generalized-torque decomposition and Level 4 random-activation / short-rollout gates do not yet pass. The model must not be described as a validated MuJoCo counterpart until those gates pass on the stated distributions.

## Claim Map

| Claim | Minimum evidence | Blocking checks |
| --- | --- | --- |
| L3 dynamic equivalence | Matched states produce matched inverse dynamics, muscle-generated generalized torques and passive generalized torques. | `inertial`, `inverse_dynamics`, `muscle_torque`, `passive_forces` |
| L3.5 generalized torque equivalence | Sparse/group/random activations produce matched independent-coordinate active generalized torque after constraint-chain projection. | `generalized_torque` |
| L4 behavior equivalence | Matched initial states and controls produce bounded short-horizon rollout drift, and contact behavior is comparable under controlled probes. | `contact`, `forward_dynamics` |

Do not use long RL rollouts as the primary equivalence proof. Long rollouts amplify small numerical differences and are useful only after short-horizon gates are stable.

## Required Adapters

Every experiment must use the same adapter layer:

- `q`: independent OpenSim coordinates plus dependent coordinates required by constraints.
- `qdot`: generalized speeds in the same coordinate order and sign convention.
- `qddot`: required only for inverse dynamics.
- root pose: explicit conversion between OpenSim root translation/Euler coordinates and MuJoCo freejoint position/quaternion.
- controls: either generalized torques or muscle activations, never both unless the experiment says so.
- contacts: explicitly disabled or explicitly enabled in both backends.
- coordinate filtering: pass/fail gates use independent coordinates first; dependent coordinates require constraint-Jacobian projection.

The adapter must reject a sample if it sets only one side of a constrained coordinate pair.

## Level 3: Dynamics Equivalence

### L3-ID-01: Static Gravity Inverse Dynamics

- Config name: `gravity_static_neutral`
- Purpose: isolate mass, COM, gravity direction, joint axes and torque sign conventions.
- Inputs: neutral pose, zero velocity, zero acceleration, contacts disabled, activations disabled.
- OpenSim computation: realize acceleration stage, run inverse dynamics on the same generalized state, export generalized torques.
- MuJoCo computation: set `qpos`, `qvel`, `qacc=0`, call inverse dynamics, export `qfrc_inverse` projected to mapped independent coordinates.
- Metrics: absolute torque error, relative torque error, sign consistency, worst coordinate.
- Outputs: `inverse_dynamics_torque_error.csv`, `diagnostics/inverse_dynamics_worst_error.csv`.
- Pass gate: max error <= `3 Nm` or relative error <= `5%`; warn above `1 Nm` or `2%`.

### L3-ID-02: No-Contact Sagittal Trajectory Inverse Dynamics

- Config name: `sagittal_sinusoid_no_contact`
- Purpose: test inertia, Coriolis and velocity-dependent terms.
- Inputs: `configs/dynamics_samples/sagittal_sinusoid.csv`, contacts disabled.
- Trajectory: smooth low-amplitude hip, knee, ankle and lumbar sinusoid with analytic `qdot` and `qddot`.
- Metrics: time-series torque RMSE, max torque error, per-coordinate correlation.
- Outputs: `inverse_dynamics_torque_error.csv`, `diagnostics/inverse_dynamics_timeseries_worst.csv`.
- Pass gate: coordinate RMSE <= `3 Nm` or `5%`; correlation >= `0.98`.

### L3-MT-01: Single Muscle Activation Sweep

- Config name: `single_muscle_activation_sweep`
- Purpose: verify muscle force generation plus Level 2 moment arms muscle by muscle.
- Inputs: neutral pose, zero velocity, activation values `[0.0, 0.25, 0.5, 1.0]`, contacts disabled.
- OpenSim computation: set one muscle activation, hold all other activations at zero, compute generalized muscle contribution.
- MuJoCo computation: set the matching actuator control/activation, evaluate actuator/tendon force and generalized contribution.
- Metrics: muscle force error, generalized torque error on mapped independent coordinates, direction consistency.
- Outputs: `muscle_generated_torque_error.csv`, `diagnostics/muscle_torque_worst_single_muscle.csv`.
- Pass gate: torque error <= `3 Nm` or `10%`; warn above `1 Nm` or `5%`.

### L3-MT-02: Muscle Group Activation

- Config name: `antagonist_group_activation`
- Purpose: verify control ordering and net torque direction for coordinated RL actions.
- Groups: hip flexors, hip extensors, knee extensors, ankle plantarflexors, and their left/right mirrors.
- Inputs: neutral pose and two non-neutral poses after L3-ID-02 is stable.
- Metrics: net generalized torque error, expected agonist/antagonist sign, top contributing muscles.
- Outputs: `muscle_generated_torque_error.csv`, `diagnostics/muscle_torque_group_error.csv`.
- Pass gate: net torque error <= `3 Nm` or `10%`; sign must match for dominant coordinates.

### L3-MT-03: Sparse Random Activation Vectors

- Purpose: catch actuator indexing mistakes that single-muscle sweeps can miss.
- Inputs: fixed random seed, 20 sparse activation vectors, 5 to 10 active muscles each, contacts disabled.
- Metrics: full independent-coordinate generalized torque RMSE and correlation.
- Outputs: `muscle_generated_torque_error.csv`, `diagnostics/muscle_torque_random_vectors.csv`.
- Pass gate: correlation >= `0.98`; max dominant-coordinate error <= `10%` or `3 Nm`.

### L3-PF-01: Passive Joint Grid

- Config name: `passive_joint_grid`
- Purpose: compare passive muscle, ligament, stiffness, damping and joint-limit forces.
- Inputs: `configs/dynamics_samples/passive_joint_grid.csv`, zero activation, zero velocity, contacts disabled.
- Metrics: total passive generalized torque error, component errors when both backends expose components.
- Outputs: `passive_force_comparison.csv`, `ligament_comparison.csv`, `diagnostics/passive_force_worst_error.csv`.
- Pass gate: passive torque error <= `2 Nm` or `10%`; warn above `0.5 Nm` or `5%`.

### L3-PF-02: Passive Velocity Sweep

- Purpose: isolate damping and force-velocity passive terms.
- Inputs: neutral and joint-limit poses, `qdot` values `[-1.0, -0.5, 0.5, 1.0] rad/s` on one coordinate at a time, zero activation.
- Metrics: damping/passive velocity torque slope, sign consistency.
- Outputs: `passive_force_comparison.csv`, `diagnostics/passive_velocity_sweep.csv`.
- Pass gate: slope correlation >= `0.98`; sign must match.

## Level 4: Behavior Equivalence

Before running Level 4, Level 3.5 must pass. The current `generalized_torque` check writes:

- `diagnostics/random_activation_torque_errors.csv`
- `diagnostics/per_coordinate_torque_decomposition.csv`
- `diagnostics/per_muscle_contribution_worst_rows.csv`
- `diagnostics/inertia_amplification_report.csv`
- `diagnostics/active_muscles_in_failing_vectors.csv`

Current strict result: Level 3.5 fails with max active generalized-torque error about `4.51 Nm`; the worst row is `semimem_r` on `knee_angle_r`. The previous much larger knee torque error was reduced by making the check constraint-chain-aware, so dependent-coordinate projection is mandatory for all future torque claims.

### L4-C-01: Contact Inventory and Parameter Review

- Config source: `contacts` in `configs/model_mapping.yaml`.
- Purpose: identify whether OpenSim and MuJoCo contact pairs have comparable geometry, friction, stiffness and damping.
- Metrics: missing contact mapping count, geometry type mismatch, friction mismatch, normal stiffness/damping mismatch.
- Outputs: `contact_model_comparison.csv`, `contact_notes.md`.
- Pass gate: all RL-relevant contacts mapped; unsupported pairs explicitly excluded.

### L4-C-02: Contact Probe Grid

- Purpose: compare normal and tangential contact behavior under controlled penetrations and sliding velocities.
- Inputs: `configs/dynamics_samples/contact_probe_grid.csv`.
- Probes: heel/toe/forefoot against ground, penetration depths `[1, 2, 5, 10] mm`, sliding speeds `[0.0, 0.1, 0.5] m/s`.
- Metrics: normal force error, friction force error, contact point, center of pressure, penetration depth.
- Outputs: `contact_behavior_comparison.csv`, `diagnostics/contact_probe_worst_error.csv`.
- Pass gate: normal force error <= `15%`, friction direction must match, contact point error <= `2 cm`.

### L4-FD-01: Passive No-Contact Rollout

- Config name: `passive_no_contact_100ms`
- Purpose: smoke-test gravity, inertia and passive forces without contact discontinuities.
- Inputs: neutral pose, zero velocity, zero activation, contacts disabled, horizon `0.1 s`, dt `0.001 s`.
- Metrics: COM drift error, root pose error, independent `q/qdot` drift.
- Outputs: `forward_dynamics_smoke_test.csv`, `diagnostics/forward_no_contact_drift.csv`.
- Pass gate: COM drift <= `20 mm`; coordinate drift <= `0.05 rad` or `10 mm`; warn above `5 mm` or `0.01 rad`.

Current executable precursor: `passive_no_contact_instant_acceleration` compares t=0 independent-coordinate `qacc` before running a long rollout. This gate now passes after adding generated OpenSim `ExpressionBasedCoordinateForce` entries named `level4_neutral_qacc_fit_*`.

The generated force fit is documented in `configs/dynamics_samples/level4_neutral_qacc_fit_forces.csv`. It reduces the neutral no-contact instant-acceleration error from the original worst case of about `4699 rad/s^2` to a max error of about `6.5e-11` over 71 evaluated independent coordinates. Locked OpenSim coordinates are skipped by the gate.

### L4-FD-02: Single Joint Torque Pulse

- Config name: `single_joint_torque_pulse_50ms`
- Purpose: validate acceleration response under matched generalized forces.
- Inputs: neutral pose, contacts disabled, 50 ms torque pulses on hip, knee and ankle.
- Metrics: `q`, `qdot`, and coordinate acceleration response error.
- Outputs: `forward_dynamics_smoke_test.csv`, `diagnostics/forward_torque_pulse_error.csv`.
- Pass gate: coordinate drift <= `0.03 rad` over 50 ms; velocity response correlation >= `0.98`.

### L4-FD-03: Matched Activation Rollout

- Config name: `matched_activation_rollout_200ms`
- Purpose: test short-horizon muscle-driven behavior after Level 3 muscle torque passes.
- Inputs: `configs/dynamics_samples/matched_activation_step.csv`, contacts disabled, horizon `0.2 s`.
- Metrics: COM drift, independent coordinate drift, muscle-tendon length drift, generalized torque drift.
- Outputs: `forward_dynamics_smoke_test.csv`, `diagnostics/forward_activation_rollout_error.csv`.
- Pass gate: COM drift <= `30 mm`; coordinate drift <= `0.05 rad`; muscle-length drift <= `5 mm`.

### L4-FD-04: Contact Drop and Standing Smoke Test

- Purpose: behavior-level contact gate after no-contact forward dynamics passes.
- Inputs: low-height foot/whole-body drop states, zero controls and simple standing activations.
- Metrics: ground reaction force, contact timing, contact impulse, COM bounce height, foot slip.
- Outputs: `contact_behavior_comparison.csv`, `forward_dynamics_smoke_test.csv`, `diagnostics/contact_rollout_error.csv`.
- Pass gate: GRF impulse error <= `20%`; contact timing error <= `10 ms`; foot slip direction must match.

## Run Order

| Stage | Runs | Go/no-go gate |
| --- | --- | --- |
| S0 adapter sanity | root conversion, coordinate order, activation order, contact enable/disable | no unmapped independent coordinate used by experiments |
| S1 Level 3 static | L3-ID-01 | torque sign and gravity terms pass before trajectory tests |
| S2 Level 3 trajectory | L3-ID-02, L3-PF-01, L3-PF-02 | inverse/passive torque gates pass |
| S3 Level 3 muscle | L3-MT-01, L3-MT-02, L3-MT-03 | muscle torque gates pass |
| S4 Level 4 no contact | L4-FD-01, L4-FD-02, L4-FD-03 | short no-contact rollouts pass |
| S5 Level 4 contact | L4-C-01, L4-C-02, L4-FD-04 | contact probes and contact rollouts pass |

If a run fails, fix the lowest-level failing cause first. For example, do not tune contact parameters while L3 inverse dynamics still has a gravity torque sign mismatch.

## Data Files

Sample input templates live in `configs/dynamics_samples/`:

- `sagittal_sinusoid.csv`: trajectory samples for inverse dynamics.
- `passive_joint_grid.csv`: passive pose grid for passive-force checks.
- `matched_activation_step.csv`: activation profile for muscle-driven forward rollout.
- `single_joint_torque_pulse.csv`: generalized torque pulse specification.
- `contact_probe_grid.csv`: prescribed contact penetration/sliding probes.

Dense production versions should be generated from these templates before enabling pass/fail gates in CI.

## Report Status Rules

Level 3 can be `passed` only when:

- `inertial`, `inverse_dynamics`, `muscle_torque` and `passive_forces` are all `passed`;
- every check writes numeric error CSVs, not only plan files;
- dependent-coordinate diagnostics are either constraint-projected or explicitly excluded from gates.

Level 4 can be `passed` only when:

- Level 3 and Level 3.5 are already `passed`;
- RL-relevant contact pairs are mapped to OpenSim contact geometries;
- the no-contact instant-acceleration gate passes under matched neutral state and zero controls;
- sparse random activation qacc passes;
- no-contact short rollout passes;
- RL action/state distribution rollout passes if RL interchangeability is claimed.

At the current state, strict Level 4 is `failed`. Contact probe grids and contact rollouts remain separate claims. Passing no-contact Level 4 does not imply contact equivalence, and none of these tests imply arbitrary long-time bitwise identity because OpenSim and MuJoCo use different integrators, constraint handling, contact semantics, and MuJoCo joint armature regularization.
