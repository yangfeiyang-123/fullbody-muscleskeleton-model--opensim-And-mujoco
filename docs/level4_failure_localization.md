# Level 4 Failure Localization

本文档记录当前 OpenSim vs MuJoCo 等价性验证中 Level 4 失败的定位结果。结论基于当前仓库的 `src/msk_equivalence` 检查，不修改 MuJoCo reference model。

## 当前结论

当前 OpenSim 模型不能声明为 Level 4 behaviorally equivalent，也不能声明为 RL-interchangeable MuJoCo counterpart。

原因不是结构、FK、body frame orientation 或 Level 3.5 主动力矩覆盖问题，而是：

1. MuJoCo `dof_armature` / generalized inertia 与 OpenSim 原生 `.osim` 动力学不等价；
2. 即使使用显式 mass-adapter diagnostic，distal / wrist / hand 坐标仍有较大 residual；
3. 这些 residual 大多发生在当前 random activation vector 没有 active mapped muscle contribution 的坐标上，因此不能通过继续调 muscle path / moment arm 来解决；
4. 直接改 OpenSim body inertia 会破坏 Level 3 inertial 语义；加 damping 已被 probe 证明不能修 instant qacc，并会恶化 rollout；提高门限不是修复。

## 已排除的问题

### FK / orientation 不是主因

新增检查：

- `orientation_audit`
- `random_fk`

验证结果：

- `orientation_audit`
  - evaluated rows: `101`
  - max orientation error: `0 rad`
  - status: `passed`
- `random_fk`
  - sample count: `500`
  - body rows: `50500`
  - marker/site rows: `1010500`
  - orientation rows: `50500`
  - max position error: `3.82e-9 m`
  - RMSE position error: `2.50e-10 m`
  - max orientation error: `1.10e-7 rad`
  - status: `passed`

因此 Level 4 失败不是 root frame、multi-joint FK、body orientation 或坐标 sign 的隐藏错误。

### Level 3.5 主动力矩覆盖不是主因

当前 Level 3.5 generalized torque:

- evaluated rows: `324`
- max torque error: `0.0983 Nm`
- RMSE: `0.00991 Nm`
- missing coverage rows: `0`
- status: `passed`

在 Level 4 worst vector `2` 中：

- active MuJoCo actuators: `ANC EDC3_left LTpT_R4_r MF_m2s_l addmagProx_l gasmed_r glmax2_r glmax3_l`
- `ankle_angle_r` active torque error: `0.00708 Nm`
- `knee_angle_r` active torque error: `0.00913 Nm`
- 多数 wrist / hand 坐标 active contributors 为 `0`

因此继续修 active muscle force / moment arm 不能解释大部分 Level 4 residual。

## Mass-Adapter Diagnostic

新增 diagnostic:

- `forward_dynamics_mass_adapter`

该检查将 OpenSim raw qacc 通过 MuJoCo with-armature generalized-force sensitivity 映射，用来判断“如果显式引入 MuJoCo generalized inertia / armature adapter，误差能否消失”。它不是 native `.osim` gate，不能替代 strict Level 4。

单向量调试命令：

```bash
PYTHONPATH=src MSK_EQUIVALENCE_SKIP_PLOTS=1 /data3/yangfeiyang/conda_envs/mujoco-opensim/bin/python -m msk_equivalence.compare \
  --osim MimicMSK_Model_opensim/MimicMSK_OpenSim.osim \
  --mjcf MimicMSK_Model_mujoco/body/myofullbody.xml \
  --mapping configs/model_mapping.yaml \
  --out /tmp/equiv_mass_adapter_v3_idx2 \
  --checks forward_dynamics_mass_adapter \
  --debug-vector-index 2
```

结果：

- evaluated rows: `71`
- max qacc error: `200.19`
- RMSE: `56.53`
- max relative error: `1.86`
- residual generalized force max abs: `13.52 Nm`
- residual generalized force RMSE: `1.82 Nm`
- status: `failed`

这说明 mass / armature adapter 可以显著降低 raw qacc error，但仍不能让 Level 4 通过。

新增 residual force 输出：

- `diagnostics/forward_dynamics_mass_adapter_residual_generalized_force.csv`
- `diagnostics/forward_dynamics_mass_adapter_residual_generalized_force_worst.csv`
- `diagnostics/forward_dynamics_mass_adapter_zero_activation_qacc.csv`
- `diagnostics/forward_dynamics_mass_adapter_zero_activation_residual_generalized_force.csv`

该 residual 是将 mass-adapter 后的 qacc residual 通过 MuJoCo with-armature sensitivity inverse 反推得到的估计广义力。它用于定位 bias / passive / correction-force 级别的剩余差异，不是 pass/fail gate。

worst residual generalized force rows:

| coordinate | qacc residual | estimated required generalized force |
| --- | ---: | ---: |
| `flex_extension` | `13.11` | `13.52 Nm` |
| `knee_angle_r` | `44.84` | `3.59 Nm` |
| `ankle_angle_l` | `197.81` | `2.92 Nm` |
| `ankle_angle_r` | `200.19` | `2.87 Nm` |
| `hip_flexion_l` | `40.58` | `2.65 Nm` |
| `hip_flexion_r` | `40.05` | `1.70 Nm` |
| `knee_angle_l` | `42.09` | `1.58 Nm` |

这些值说明：一些坐标上的 qacc residual 很大，但对应 generalized force residual 只有数 Nm 量级；这是小惯量 / 高 sensitivity 坐标对 residual force 的放大。

### Zero-Baseline Runtime Adapter Probe

新增 diagnostic：

- `diagnostics/forward_dynamics_mass_adapter_zero_baseline_corrected_qacc.csv`

该 probe 不修改 `.osim`，而是在 `mass_adapter` 之后显式施加 zero-activation residual generalized-force baseline，用来判断 runtime adapter 路线是否有价值。它不是 native OpenSim gate。

同一最坏 random activation vector `2` 的结果：

- mass-adapter qacc max error: `200.19`
- mass-adapter qacc RMSE: `56.53`
- zero-baseline corrected qacc max error: `10.23`
- zero-baseline corrected qacc RMSE: `1.24`
- zero-baseline corrected max relative error: `0.54`

worst absolute row:

| coordinate | corrected qacc error | corrected OpenSim qacc | MuJoCo qacc |
| --- | ---: | ---: | ---: |
| `subtalar_angle_r` | `10.23` | `-245.07` | `-255.31` |

worst relative row:

| coordinate | corrected qacc error | relative error | corrected OpenSim qacc | MuJoCo qacc |
| --- | ---: | ---: | ---: | ---: |
| `pm4_flexion_r` | `0.0179` | `0.541` | `-0.0152` | `-0.0331` |

解释：

zero-baseline runtime adapter 对 vector `2` 非常有效，说明 Level 4 剩余误差中有一大部分是稳定 baseline/bias residual，而不是 active muscle geometry 问题。它仍不能证明 native `.osim` 通过，因为该修正是显式 runtime adapter；此外 max relative error 仍失败，且尚未经过全部 random vectors、no-contact rollout 和 RL action/state distribution rollout 验证。

全量 20 个 sparse random activation vectors 的结果：

- mass-adapter qacc max error: `302.18`
- mass-adapter qacc RMSE: `59.28`
- zero-baseline corrected qacc max error: `233.90`
- zero-baseline corrected qacc RMSE: `13.39`
- zero-baseline corrected max relative error: `1.33`

全量 worst rows：

| vector | coordinate | corrected qacc error | corrected OpenSim qacc | MuJoCo qacc |
| ---: | --- | ---: | ---: | ---: |
| `17` | `mp_flexion_r` | `233.90` | `125.13` | `359.03` |
| `12` | `cmc_flexion_r` | `198.53` | `-248.28` | `-49.75` |
| `14` | `cmc_flexion_l` | `169.11` | `-186.29` | `-17.19` |
| `12` | `cmc_abduction_r` | `139.17` | `-34.33` | `104.85` |

因此，zero-baseline adapter 只能解释一部分稳定 bias residual；完整随机激活集里仍存在 activation-dependent residual，主要集中在 thumb/finger small-inertia coordinates。下一步如果走 runtime adapter 路线，必须继续分解这些手部坐标的 active muscle force、moment arm、muscle equilibrium / activation floor 和 MuJoCo actuator force realization，不能只依赖 zero-baseline 修正。

## Native Path/Wrap Repair Probes

对手部 worst rows 做了两个 `/tmp` 临时 OpenSim 探针；这些探针没有写入仓库模型。

### FPL PathPoint Reversion Probe

将 `FPL_p7/FPL_p8` 和 `FPL_left_p7/FPL_left_p8` 临时回退到 MuJoCo site 坐标：

- `FPL_p7`: `0.00656255 -0.023 -0.0096` -> `0.0021 -0.023 -0.0096`
- `FPL_p8`: `-0.00122648 -0.0184 -0.0051` -> `-0.0045 -0.0184 -0.0051`
- left side mirrored

结果：vector `12` 的 `mp_flexion_r` torque error 从约 `0.000155 Nm` 恶化到 `0.416 Nm`。该修复不能采用。

### EPB MPthumb Sidesite Probe

MuJoCo `EPB_tendon` 使用 `MPthumb_wrap` + `sidesite=MPthumb_site_EPB_side`，而 OpenSim 只能用 `quadrant` 近似。临时加入显式 `EPB_p7_sidesite` 并把 `EPB_wrap1` range 顺延后：

- `mp_flexion_r` EPB moment-arm error 从 `0.00278 m` 降到 `0.000567 m`
- `mp_flexion_r` active torque error 从 `0.0458 Nm` 降到 `0.00935 Nm`
- 但 raw qacc 从约 `389` 变成 `1028`，Level 4 动力学反而恶化
- muscle length max error 从 `0.00291 m` 恶化到 `0.01047 m`
- EPB zero-command passive force 从 `0.446 N` 变成 `0.897 N`，而 MuJoCo 为 `0.440 N`

该 probe 说明 EPB 的 MuJoCo sidesite wrapping 语义确实影响 hand Level 4 residual，但简单把 sidesite 变成 OpenSim PathPoint 会扰动 muscle length、passive force baseline 和 qacc，不能作为保守模型修复。

## Zero-Activation Baseline

为了确认 residual 是否由 random activation 触发，新增了 zero-activation no-contact mass-adapter diagnostic。该 diagnostic 关闭 level4 correction-force probes，设置 OpenSim muscle activations 和 MuJoCo controls 为 0，然后使用同一个 mass adapter 进行 qacc 对比。

结果：

- zero-activation max qacc error: `200.52`
- zero-activation RMSE qacc error: `56.49`
- zero-activation residual generalized force max abs: `9.80 Nm`
- zero-activation residual generalized force RMSE: `1.37 Nm`

zero-activation worst rows:

| coordinate | qacc residual | estimated required generalized force |
| --- | ---: | ---: |
| `flex_extension` | `13.10` | `9.80 Nm` |
| `ankle_angle_l` | `197.71` | `2.90 Nm` |
| `ankle_angle_r` | `200.52` | `2.90 Nm` |
| `knee_angle_r` | `44.72` | `2.49 Nm` |
| `knee_angle_l` | `42.13` | `1.97 Nm` |
| `hip_flexion_r` | `41.70` | `1.44 Nm` |
| `hip_flexion_l` | `40.61` | `1.38 Nm` |

这与 random activation vector `2` 的 residual pattern 高度一致。因此当前 Level 4 剩余失败主要是 passive / bias / baseline dynamics 差异，而不是 random activation 下 active muscle torque 失配。

## Fitted Correction-Force Audit

当前 OpenSim 文件中存在 71 个 `level4_neutral_qacc_fit_*` `ExpressionBasedCoordinateForce`。这些项是早期为中性点瞬时加速度拟合出来的 coordinate force，不是 MuJoCo armature / solver / passive dynamics 的原生等价表达。

新增检查：

- `level4_correction_force_audit`

调试命令：

```bash
PYTHONPATH=src MSK_EQUIVALENCE_SKIP_PLOTS=1 /data3/yangfeiyang/conda_envs/mujoco-opensim/bin/python -m msk_equivalence.compare \
  --osim MimicMSK_Model_opensim/MimicMSK_OpenSim.osim \
  --mjcf MimicMSK_Model_mujoco/body/myofullbody.xml \
  --mapping configs/model_mapping.yaml \
  --out /tmp/equiv_correction_force_audit_idx2_v2 \
  --checks level4_correction_force_audit \
  --debug-vector-index 2
```

结果：

- probe count: `71`
- evaluated rows: `142`
- improved rows: `134`
- degraded rows: `8`
- zero activation max qacc error with correction: `125.57`
- zero activation RMSE with correction: `29.74`
- zero activation max qacc error without correction: `5549.87`
- random activation vector 2 max qacc error with correction: `48611.33`
- random activation vector 2 RMSE with correction: `6617.85`
- random activation vector 2 max qacc error without correction: `54168.24`
- status: `failed`

解释：

这些常数 correction force 对中性点 zero-activation 有明显帮助，但 held-out random activation 仍然失败，且仍有 8 行被 correction force 变差。因此它们不能作为 native OpenSim Level 4 等价性的证据，也不能继续通过扩大这类常数力来制造通过。若以后需要 correction force，必须单独声明为 runtime dynamics adapter，并用 held-out random qacc、no-contact rollout 和 RL action/state rollout 验证。

## Worst Residual Pattern

在 `debug-vector-index 2` 下，mass-adapter 后仍超过 `100` 的坐标包括：

| coordinate | mass-adapter qacc error | active contributors | active torque error |
| --- | ---: | ---: | ---: |
| `ankle_angle_r` | `200.19` | `1` | `0.00708 Nm` |
| `ankle_angle_l` | `197.81` | `0` | `0` |
| `deviation_r` | `118.42` | `0` | `0` |
| `deviation_l` | `112.94` | `0` | `0` |
| `cmc_flexion_r` | `103.73` | `0` | `0` |
| `cmc_flexion_l` | `101.93` | `0` | `0` |

超过 `50` 的坐标共有 `18` 个，主要集中在:

- wrist: `pro_sup_*`, `deviation_*`, `flexion_*`
- thumb / finger: `cmc_*`, `mcp*_abduction`, `md3_flexion_*`
- ankle: `ankle_angle_*`

这些坐标中的多数在该 vector 中没有 active mapped muscle contribution，因此 residual 更可能来自：

- passive force / limit force 语义差异；
- correction force 语义差异；
- generalized inertia cross-coupling；
- OpenSim Thelen2003 muscle equilibrium / minimum fiber length behavior；
- MuJoCo actuator/passive bias 与 OpenSim force realization 的语义差异。
- zero-activation baseline dynamics 差异。

## 不建议的修复

以下做法当前不应采用：

1. 提高 Level 4 qacc / rollout thresholds。
2. 删除 distal / wrist / hand 坐标来制造 full-body strict pass。
3. 修改 MuJoCo reference model。
4. 继续盲调 muscle path / wrap object，因为 Level 3.5 active torque 已基本对齐。
5. 把 MuJoCo armature 硬塞进 OpenSim body inertia，因为会破坏 Level 3 inertial/body parameter 语义。
6. 用 fitted correction force 直接压 random activation / rollout failure，除非单独声明为 runtime adapter，并做 held-out validation。

## 下一步修复路线

### 路线 A: Native `.osim` strict Level 4

当前证据不支持这条路线可以在不破坏 Level 0-3 的情况下完成。OpenSim 原生 `.osim` 缺少 MuJoCo coordinate-level armature 的直接等价机制。

### 路线 B: OpenSim + runtime dynamics adapter

这是目前最可解释的工程路线：

1. 保持 `.osim` 的 Level 0-3 / Level 3.5 等价；
2. 显式实现 coordinate-level generalized inertia / armature adapter；
3. 对 adapter 做 random activation qacc、no-contact rollout、RL action/state distribution rollout；
4. 报告中声明模型是 `OpenSim model + explicit MuJoCo-dynamics adapter counterpart`，而不是 native `.osim` behaviorally equivalent。

### 路线 C: RL-actionable subspace

如果 RL policy 实际不控制或不观测 distal hand/finger 坐标，应单独定义 `rl_actionable` subspace：

1. full-body strict report 必须保留失败；
2. 只在 RL state/action 维度上验证 qacc 和 rollout；
3. 报告中不能把 RL-subspace pass 扩大成 full-body strict pass。

## 当前 Claim

当前可以声称：

- Level 0-3 selected checks 通过；
- Level 3.5 generalized torque decomposition 通过；
- FK / orientation / random legal-pose kinematics 通过；
- strict native Level 4 失败；
- mass/armature 是主因之一，但不是全部；
- 不能声称 native OpenSim `.osim` 是 validated MuJoCo counterpart 或 RL-interchangeable model。
