# Dynamics Equivalence Experiments

The goal is to prove that OpenSim and MuJoCo are not only visually or kinematically similar, but produce the same dynamics under matched state and control inputs.

## Required Gate Order

1. Level 1 kinematics must pass: matched poses must produce the same body, marker/site and whole-body COM positions.
2. Level 2 muscle geometry must pass: muscle-tendon lengths and independent-coordinate moment arms must match.
3. Level 3 dynamics can then be trusted: inverse dynamics, muscle-generated torque and passive forces.
4. Level 4 short rollouts can be used as behavior smoke tests.

## State Adapter

Every experiment must pass through a shared state adapter:

- `q`: mapped independent coordinates plus dependent coordinates required by constraints;
- `qdot`: matched generalized speeds;
- `qddot`: only for inverse dynamics;
- root pose: explicit conversion between OpenSim root Euler coordinates and MuJoCo freejoint position/quaternion;
- contacts: explicitly disabled or enabled in both engines.

The adapter must reject samples that set only one side of a constrained coordinate pair.

## Experiment Set

The configured plan is stored in `configs/model_mapping.yaml` under `dynamics_experiments`.

### Inverse Dynamics

- `gravity_static_neutral`: zero velocity and acceleration, contact disabled. This isolates gravity, mass, COM, joint axes and torque sign conventions.
- `sagittal_sinusoid_no_contact`: short smooth trajectory for hip, knee, ankle and lumbar coordinates. This tests inertia, Coriolis and velocity-dependent terms.

Compare generalized torques on independent coordinates. Dependent coordinates should be validated through the constraint Jacobian, not by direct one-to-one torque comparison.

### Muscle-Generated Torque

- `single_muscle_activation_sweep`: activate one muscle at a time at several activation values.
- `antagonist_group_activation`: activate expected agonist/antagonist groups and compare net generalized torque direction and magnitude.

This gate should use the same activation ordering, activation dynamics assumptions and muscle parameter extraction on both backends.

### Passive Forces

- `passive_joint_grid`: zero activation, zero velocity, grid over representative joint limits.

Compare passive generalized torque from passive muscle force, ligament force, damping, stiffness and joint-limit terms separately when the backend exposes them. If a backend only exposes total passive torque, record that limitation in the report.

### Forward Dynamics

- `passive_no_contact_100ms`: zero controls, no contact, 100 ms.
- `single_joint_torque_pulse_50ms`: matched generalized torque pulse, no contact, 50 ms.
- `matched_activation_rollout_200ms`: matched activation profile, no contact, 200 ms.

Forward rollout thresholds must be short-horizon thresholds. Long rollouts are not a strict equivalence proof because small numerical differences can diverge.

## Suggested Acceptance Thresholds

- Inverse dynamics torque: warn at 2 percent relative error or 1 Nm absolute error; fail at 5 percent or 3 Nm.
- Muscle-generated torque: warn at 5 percent relative error or 1 Nm; fail at 10 percent or 3 Nm.
- Passive torque: warn at 5 percent relative error or 0.5 Nm; fail at 10 percent or 2 Nm.
- Forward COM drift: warn at 5 mm over 100 ms; fail at 20 mm.
- Forward coordinate drift: warn at 0.01 rad or 2 mm; fail at 0.05 rad or 10 mm.

These thresholds should be tightened after Level 2 geometry passes and backend integration details are confirmed.
