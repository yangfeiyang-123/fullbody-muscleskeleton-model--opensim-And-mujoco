# Completion Audit: OpenSim vs MuJoCo Counterpart Goal

本文档记录当前目标的完成审计。目标是：在不修改 MuJoCo reference model 的前提下，修复 OpenSim 模型，使 Level 3.5 和 Level 4 通过，同时不让 Level 0-3 变差，并确认 Level 3.5 / Level 4 验证本身是否合理。

## Objective As Deliverables

| Requirement | Required Evidence | Current Evidence | Status |
| --- | --- | --- | --- |
| Do not modify MuJoCo model | Git diff must not include MuJoCo model files | `git status --short` shows no modified `MimicMSK_Model_mujoco/*` paths | Satisfied |
| Level 0-3 must not regress | Re-run or inspect affected gates after OpenSim edits | `/tmp/equiv_0_35_after_wrapfix/summary.json`: kinematics, joint sweep, muscle length, moment arm, inertial, inverse dynamics, muscle torque and passive forces are passed under configured gates | Satisfied for current checked gates |
| Level 3.5 must pass | Generalized torque decomposition pass, no hidden missing coverage | `/tmp/equiv_0_35_after_wrapfix/summary.json`: `generalized_torque` passed, 324 rows, max torque error `0.0983299002645679 Nm`, RMSE `0.009914946676079605 Nm`, missing coverage rows `0` | Satisfied |
| Confirm Level 3.5 is reasonable | Diagnostic must decompose source by coordinate/muscle and expose amplification | `generalized_torque.py` writes random activation torque errors, per-coordinate decomposition, per-muscle contribution rows, inertia amplification, missing coverage and active-muscle diagnostics | Satisfied for sparse random activation at neutral pose; not a rollout guarantee |
| Level 4 strict native OpenSim must pass | Native `.osim` random activation qacc and no-contact rollout pass configured gates | `/tmp/equiv_l4_after_wrapfix/summary.json`: `forward_dynamics` failed, random activation max qacc error `48611.32665781625`, RMSE `1852.8581842674248`, rollout failed with extremely large q/qdot/qacc errors | Not satisfied |
| Confirm Level 4 validation is reasonable | Must separate raw strict gate from diagnostics and not hide failure | `forward_dynamics.py`, `forward_dynamics_mass_adapter.py`, `level4_correction_force_audit.py`, and `docs/level4_failure_localization.md` separate strict raw failure from diagnostic adapters | Satisfied |
| Do not pass by loosening thresholds | Thresholds must not be raised to hide errors | No threshold-loosening repair is used; failed diagnostics remain failed | Satisfied |
| Do not hide coordinate/muscle extremes | Worst-row reports must preserve failing distal/hand coordinates | Level 4 reports expose `mtp_angle_*`, `ankle_angle_*`, `mp_flexion_r`, `cmc_flexion_*`, `cmc_abduction_*`, wrist/finger rows | Satisfied |
| Correction forces must be held-out validated | Existing fitted forces must fail if they only fit neutral point | `/tmp/equiv_correction_force_audit_idx2_v2/summary.json`: `level4_correction_force_audit` failed; random activation with correction max qacc error `48611.32665781625` | Satisfied as a guard; not a pass |
| Runtime adapter route, if used, must be explicit | Adapter diagnostics cannot be reported as native `.osim` pass | `/tmp/equiv_mass_adapter_baseline_all/summary.json`: mass adapter remains diagnostic-only and failed; zero-baseline corrected all-vector max qacc error `233.90106045116394`, RMSE `13.385717151907256` | Not a native pass |

## Current Metrics

Current successful Level 0-3 / 3.5 evidence from `/tmp/equiv_0_35_after_wrapfix/summary.json`:

- `inertial`: max segment mass error `1.963762485956977e-12 kg`
- `kinematics`: max error `9.615449302236795e-08 m`
- `muscle_length`: max error `0.002914578108841104 m`
- `moment_arm`: direct independent max error `0.09672988560693335 m`, correlation `0.9832685760537869`
- `inverse_dynamics`: max torque error `7.389644451905042e-07 Nm`, RMSE `1.6140479708244954e-07 Nm`
- `muscle_torque`: max active force error `5.135223717099393 N`, RMSE `0.325235616867152 N`
- `passive_forces`: max passive force error `8.30331195063927 N`, RMSE `0.9967444451610189 N`
- `generalized_torque`: max torque error `0.0983299002645679 Nm`, RMSE `0.009914946676079605 Nm`

Current Level 4 strict failure evidence from `/tmp/equiv_l4_after_wrapfix/summary.json`:

- neutral no-contact instant qacc max error `125.57169527414555`
- random activation qacc max error `48611.32665781625`
- random activation qacc RMSE `1852.8581842674248`
- rollout max q error `31577701756.01116`
- rollout max qdot error `31577801231165.75`
- rollout qacc max error `3.1577901727051244e+16`

Current adapter diagnostics:

- mass-adapter all-vector max qacc error `302.18231840677544`
- mass-adapter all-vector RMSE `59.27625219752474`
- zero-baseline corrected all-vector max qacc error `233.90106045116394`
- zero-baseline corrected all-vector RMSE `13.385717151907256`
- zero-baseline corrected all-vector max relative error `1.3274257404423138`

## Rejected Native Repair Probes

The following probes were run on `/tmp` copies only and were not applied to the repository OpenSim model:

1. FPL PathPoint reversion to MuJoCo site positions.
   - Rejected because vector `12` `mp_flexion_r` torque error worsened from about `0.000155 Nm` to `0.416 Nm`.

2. EPB MPthumb sidesite PathPoint insertion.
   - Improved EPB `mp_flexion_r` moment arm error from `0.00278 m` to `0.000567 m`.
   - Rejected because raw qacc worsened from about `389` to `1028`, muscle length max error worsened from `0.00291 m` to `0.01047 m`, and EPB passive force changed from `0.446 N` to `0.897 N` versus MuJoCo `0.440 N`.

These probes show that simple hand/thumb path edits can improve an active moment arm while breaking Level 2/3/passive dynamics. They are not conservative fixes.

## Completion Decision

The goal is not complete. Level 3.5 is currently passed and the verification is reasonable for the tested sparse random activation distribution, but strict native Level 4 is still failed. The current evidence does not support another safe local `.osim` path/wrap edit that would pass Level 4 without risking Level 0-3 or Level 3.5.

The next productive path requires a decision:

1. Continue a high-risk native OpenSim joint optimization over hand/thumb path geometry, tendon slack length, passive baseline and qacc, with full Level 0-3/3.5 revalidation after every accepted step.
2. Switch the claim to `OpenSim model + explicit runtime dynamics adapter`, then implement held-out validation for random activation qacc, no-contact short rollout, RL action/state distribution rollout and contact rollout if claimed.

Until one of those paths is explicitly chosen and passes its own gates, the model must be reported as not Level 4 behaviorally equivalent and not RL-interchangeable with the MuJoCo reference.

## Runtime Adapter Implementation Update

The project now implements the second route: `OpenSim model + explicit runtime dynamics adapter`. This is a separate claim from pure native `.osim` Level 4 equivalence.

Implemented artifacts:

- `configs/adapter_validation.yaml`
- `src/msk_equivalence/adapters/runtime_dynamics.py`
- `src/msk_equivalence/checks/forward_dynamics_adapter.py`
- `src/msk_equivalence/checks/adapter_no_contact_rollout.py`
- `src/msk_equivalence/checks/rl_distribution_rollout.py`
- `compare.py --adapter-config`
- report separation between `Level 4 Native` and `Level 4 Adapter`

Current adapter evidence:

- Core Level 0-3.5 after adapter code changes: `/tmp/equiv_core_after_adapter/summary.json`
  - Level 0: passed
  - Level 1: passed
  - Level 2: passed
  - Level 3: passed
  - Level 3.5: passed
- Runtime adapter qacc with single-muscle basis calibration and held-out random activation test:
  - source rows: `/tmp/equiv_adapter_qacc_basis/forward_dynamics_adapter_qacc.csv`
  - current config status recomputation: passed
  - held-out test max absolute qacc error: `1.8658530581646033`
  - held-out test RMSE qacc error: `0.13561883795612037`
  - relative gate uses `qacc_relative_abs_error_floor=5.0`; all rows above that floor satisfy the relative gate
- Runtime adapter no-contact rollout:
  - `/tmp/equiv_adapter_rollout_4ms/summary.json`
  - horizon: `0.004 s`
  - max q error: `0.0032743935541762817`
  - max qdot error: `1.6376729623030868`
  - status: warning, because qdot exceeds the warning threshold but remains below the failure threshold
  - longer `0.05 s` rollout failed (`/tmp/equiv_adapter_rollout/summary.json`), showing the current adapter is local around the neutral calibration state and is not yet a state-dependent rollout adapter
- RL distribution rollout:
  - `/tmp/equiv_rl_distribution_gate/summary.json`
  - status: not evaluated
  - reason: no MuJoCo policy rollout dataset is configured

Completion decision after adapter implementation:

The implementation path is in place and the held-out qacc gate is effectively solved under the current adapter validation config, but the full revised goal is not complete. The remaining blockers are:

1. Upgrade the current neutral-state adapter into a state-dependent adapter before claiming nontrivial no-contact rollout equivalence beyond one or two simulation steps.
2. Provide or generate a MuJoCo policy rollout dataset and implement full OpenSim+adapter replay for `rl_distribution_rollout`.
3. Keep contact equivalence explicitly unclaimed unless a separate contact rollout gate is implemented and passed.
