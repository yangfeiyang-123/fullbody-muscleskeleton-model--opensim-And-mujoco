# Dynamics Sample Inputs

These files are templates for Level 3 and Level 4 OpenSim/MuJoCo equivalence experiments.

They define the columns and small representative samples expected by future dynamics adapters. Production runs should generate dense trajectories from the same schema.

## Files

- `sagittal_sinusoid.csv`: inverse-dynamics trajectory samples.
- `passive_joint_grid.csv`: passive-force pose samples.
- `matched_activation_step.csv`: activation profile for short muscle-driven rollouts.
- `single_joint_torque_pulse.csv`: generalized torque pulse schedule.
- `contact_probe_grid.csv`: contact behavior probes.

All coordinate names use OpenSim coordinate names from `configs/model_mapping.yaml`. The adapter is responsible for mapping these values into MuJoCo `qpos/qvel/qacc`.
