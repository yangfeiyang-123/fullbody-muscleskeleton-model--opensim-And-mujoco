# RL Distribution Dataset Requirement

本文档说明为了验证 `OpenSim model + explicit runtime dynamics adapter` 是否可以在 RL 训练中等价于 MuJoCo reference，需要寻找或生成什么 dataset。

这里的 dataset 不是 mocap 数据，也不是 retarget 后的参考动作轨迹。它必须是 **MuJoCo reference model 在真实 RL 环境中，由 policy rollout 产生的状态、动作和动力学响应数据**。

## 目标

当前工程已经可以验证：

- Level 0: topology / conventions
- Level 1: kinematics
- Level 2: muscle geometry
- Level 3: inertial / inverse dynamics / muscle torque / passive force
- Level 3.5: generalized torque decomposition
- Level 4 Adapter: held-out random activation qacc

但要支持“OpenSim+adapter 在 RL 训练中可以替代 MuJoCo”的结论，还需要验证：

1. RL policy 实际访问到的 `qpos/qvel/action/ctrl` 分布；
2. 同一批 state/action 下，OpenSim+adapter 的单步 qacc 是否接近 MuJoCo；
3. 同一批 state/action 下，短时 no-contact rollout 是否不快速漂移；
4. observation / reward distribution 是否和 MuJoCo 一致；
5. 如果声称 contact 等价，还必须有单独的 contact rollout dataset 和 contact gate。

因此，需要一个从 MuJoCo RL 环境中直接 dump 的 policy rollout dataset。

## 不能替代的 dataset

以下数据不能单独用于证明 RL 训练等价：

- AMASS / SMPL / mocap 原始轨迹；
- retarget 后的人体参考轨迹；
- 只包含 `qpos/qvel`、没有 policy action 的轨迹；
- 只包含随机 activation、没有真实 policy 分布的数据；
- 只在 neutral pose 或单关节 sweep 上采样的数据；
- OpenSim 自己跑出来的数据；
- 修改过 MuJoCo model 后跑出的数据。

这些数据可以作为辅助诊断或补充覆盖，但不能替代 MuJoCo policy rollout dataset。

## Reference Model 和环境

dataset 必须来自当前仓库中的 MuJoCo reference model：

```text
MimicMSK_Model_mujoco/body/myofullbody.xml
```

OpenSim 模型只作为 replay / comparison target，不能用于生成 reference rollout。

生成 dataset 时需要记录完整环境配置，包括：

- MuJoCo XML 路径；
- MuJoCo XML 文件 hash；
- RL 环境名称 / task 名称；
- policy checkpoint 路径；
- policy checkpoint hash；
- observation 归一化参数；
- reward 配置；
- action scaling / clipping 配置；
- timestep；
- frame skip；
- integrator；
- solver 参数；
- gravity；
- contact 是否启用；
- reset distribution；
- reference trajectory source，如果 reward 依赖 mimic trajectory；
- `configs/model_mapping.yaml` 的 hash。

如果这些配置缺失，后续验证很难判断误差来自模型本身、adapter、环境配置还是 action/observation 语义不一致。

## 最小可接受格式

推荐保存为 `.npz`。当前 `rl_distribution_rollout` checker 最低要求能找到以下数组：

```text
qpos      [E, T + 1, nq] 或 [N + 1, nq]
qvel      [E, T + 1, nv] 或 [N + 1, nv]
actions   [E, T, action_dim] 或 [N, action_dim]
```

其中：

- `E` 是 episode 数；
- `T` 是每个 episode 的 step 数；
- `N` 是 flatten 后的 transition 数；
- `nq` 必须等于 MuJoCo model 的 `model.nq`；
- `nv` 必须等于 MuJoCo model 的 `model.nv`；
- `action_dim` 必须等于 RL policy 输出动作维度。

这个最小格式只能让 checker 开始识别 dataset。它不足以支撑最终 RL equivalence claim。

## 推荐完整格式

为了严格验证 RL 训练等价，建议 dataset 至少包含：

```text
qpos              [E, T + 1, nq]
qvel              [E, T + 1, nv]
qacc              [E, T, nv]
actions           [E, T, action_dim]
ctrl              [E, T, nu]
obs               [E, T, obs_dim]
rewards           [E, T]
dones             [E, T]
episode_ids       [E]
step_ids          [E, T]
seeds             [E]
```

强烈建议同时包含：

```text
time              [E, T]
dt                scalar
frame_skip        scalar
terminated        [E, T]
truncated         [E, T]
reward_terms      dict-like object 或独立数组
phase             [E, T]，如果 mimic task 使用 phase
traj_ids          [E] 或 [E, T]，如果 mimic task 使用 reference trajectory
target_qpos       [E, T, nq]，如果 reward 依赖 reference qpos
target_qvel       [E, T, nv]，如果 reward 依赖 reference qvel
site_xpos         [E, T, nsite, 3]，如果 observation/reward 使用 site
body_xpos         [E, T, nbody, 3]，如果 observation/reward 使用 body position
body_xquat        [E, T, nbody, 4]，如果 observation/reward 使用 body orientation
contact_forces    [E, T, ncontact_features]，如果声称 contact 等价
```

如果 `qacc` 无法直接从环境中记录，可以后处理计算，但最好直接保存 MuJoCo `data.qacc`，因为它是 Level 4 单步动力学验证的核心 reference。

## 名称和顺序信息

必须保存 name/order metadata，否则后续很容易比较错坐标或 actuator。

推荐保存为 `.npz` 中的字符串数组，或旁边的 `.json`：

```text
qpos_names
qvel_names
joint_names
body_names
site_names
actuator_names
action_names
ctrl_names
observation_names
reward_term_names
```

最低要求：

- `qpos` 的列顺序必须和 MuJoCo `model.qpos` 一致；
- `qvel` / `qacc` 的列顺序必须和 MuJoCo `model.qvel` / `model.qacc` 一致；
- `ctrl` 的列顺序必须和 MuJoCo `model.actuator` 一致；
- `actions` 必须说明如何映射到 `ctrl`；
- 如果 action 经过 scaling、clipping、muscle activation dynamics 或 control filtering，必须保存实际施加到 MuJoCo 的 `ctrl`。

对当前任务来说，`ctrl` 比 raw policy `actions` 更重要。因为 OpenSim+adapter 要复现的是 MuJoCo 实际接收到的 actuator command，而不是 policy 网络原始输出。

## Action / Control 语义

dataset 需要明确区分：

- `actions`: policy 输出；
- `ctrl`: 经过 action scaling、clipping、muscle activation dynamics、control filter 后，实际写入 MuJoCo `data.ctrl` 的值；
- `activation`: 如果环境显式模拟 activation state，需要保存 activation state；
- `excitation`: 如果 policy 输出的是 muscle excitation，也需要单独保存。

如果当前环境中 `actions == ctrl`，也应该同时保存两份，或在 metadata 中明确说明二者完全相同。

需要记录：

```text
action_low
action_high
ctrl_low
ctrl_high
action_scale
action_bias
clip_before_scale
clip_after_scale
control_timestep
activation_dynamics_enabled
activation_time_constant
deactivation_time_constant
```

OpenSim `Thelen2003Muscle` 有 minimum activation floor，因此 zero-action / passive condition 的语义可能和 MuJoCo 不完全相同。dataset 必须保留实际 `ctrl` 或 activation，不能只保存抽象 action。

## 采样分布要求

dataset 应覆盖 policy 在训练或评估时真实访问到的状态动作分布。

建议至少包含三类 rollout：

1. **训练中期 policy rollout**
   - 用于覆盖不稳定、探索性强的动作分布；
   - 可以暴露极端 muscle group 或 finger/toe coordinate 的问题。

2. **最终 policy rollout**
   - 用于验证最终训练行为是否可迁移；
   - 应使用确定性评估和随机采样两种模式，如果训练使用 stochastic policy。

3. **reset / early episode rollout**
   - 用于覆盖刚 reset 后的姿态、速度和接触状态；
   - 很多动力学误差会在 episode 前几十步放大。

推荐规模：

```text
minimum smoke test:
  episodes: 8-16
  steps per episode: 200-500

recommended validation:
  episodes: 64-256
  steps per episode: 500-2000

stress / publication-level validation:
  episodes: 256+
  include multiple seeds
  include multiple policy checkpoints
```

对于当前工程，先生成一个 `minimum smoke test` 就能用于打通 checker；最终 claim 至少需要 `recommended validation`。

## Train / Validation / Test Split

如果 dataset 会用于拟合 state-dependent adapter，必须保留 held-out test。

推荐 split：

```text
train: 60-70%
val:   10-20%
test:  20%
```

split 应该按 episode 划分，而不是随机打散单个 timestep。否则同一个 episode 中高度相关的相邻状态会同时进入 train 和 test，导致 held-out 指标虚高。

必须保存：

```text
split_episode_ids_train
split_episode_ids_val
split_episode_ids_test
split_seed
```

最终报告只能用 held-out test 指标支撑 claim。

## Contact 和 No-contact 分离

当前项目应把 no-contact dynamics 和 contact dynamics 分开验证。

dataset 至少需要能区分：

```text
contact_enabled
contact_count
contact_body_pairs
contact_forces 或 contact_impulses
```

如果目标只是证明 no-contact RL subspace 等价，必须在报告中明确：

- contact disabled；
- contact rollout 未声明；
- 该结论不保证 foot-ground / hand-object / self-contact 行为一致。

如果声称 contact rollout 等价，则需要单独 contact dataset：

- 包含落地、支撑、离地、滑移、碰撞恢复；
- 保存 contact pair、normal force、friction force；
- 比较 contact timing、contact impulse、post-impact qvel。

contact dataset 不应该和 no-contact dataset 混在一个总指标里直接平均，否则会隐藏 contact failure。

## 质量检查

找到或生成 dataset 后，先做以下检查：

1. 数组 shape 是否一致；
2. 所有数值是否 finite；
3. `qpos.shape[-1] == model.nq`；
4. `qvel.shape[-1] == model.nv`；
5. `qacc.shape[-1] == model.nv`，如果存在；
6. `ctrl.shape[-1] == model.nu`；
7. `actions.shape[-1] == policy action_dim`；
8. episode 内时间是否连续；
9. `qpos[:, 1:]` 是否由 `qpos[:, :-1] + qvel/action` rollout 得到，而不是外部 reference 轨迹；
10. 是否包含 early termination；
11. 是否包含 reset state；
12. 是否记录了 policy checkpoint 和环境配置。

任何一项失败，都应该先修 dataset，不要进入 OpenSim+adapter 验证。

## 推荐文件组织

建议将 dataset 放在仓库外或大文件目录中，不直接提交 Git：

```text
data/rl_rollouts/
  mujoco_myofullbody_policy_<policy_id>_<date>/
    rollout.npz
    metadata.json
    README.md
```

在本仓库中只记录路径和 hash，例如：

```yaml
rl_distribution:
  source: mujoco_policy
  dataset: /data3/yangfeiyang/WorkSpace/musclemimic/data/rl_rollouts/mujoco_myofullbody_policy_xxx/rollout.npz
  metadata: /data3/yangfeiyang/WorkSpace/musclemimic/data/rl_rollouts/mujoco_myofullbody_policy_xxx/metadata.json
```

对应配置位置：

```text
configs/adapter_validation.yaml
```

## 对 GPT / 工程执行者的寻找目标

如果让 GPT 或其他工程师去“寻找 dataset”，目标不是搜索任意 `.npz`，而是确认是否已经存在以下数据：

1. MuJoCo myofullbody RL training / evaluation rollout；
2. 包含 `qpos`、`qvel`、`actions`；
3. 最好包含 `ctrl`、`qacc`、`obs`、`rewards`、`dones`；
4. 能对应当前 `MimicMSK_Model_mujoco/body/myofullbody.xml`；
5. 能对应某个明确的 policy checkpoint；
6. 有环境配置、action scaling、observation normalization；
7. 数据来自 MuJoCo simulation，而不是 OpenSim 或 mocap reference。

如果仓库里没有这种 dataset，就需要新增一个 rollout dump 脚本，从当前 RL 环境中加载 MuJoCo model 和 policy checkpoint，执行 policy rollout，并保存上述字段。

## 最终验收标准

dataset 合格后，才能继续实现或运行完整 `rl_distribution_rollout` gate。最终报告应至少包含：

- dataset path / hash；
- model XML hash；
- policy checkpoint hash；
- episode 数；
- transition 数；
- train / val / test split；
- state distribution summary；
- action / ctrl distribution summary；
- per-coordinate qacc error；
- per-coordinate short rollout q/qdot error；
- observation error；
- reward error；
- worst coordinate / worst muscle group diagnostics；
- no-contact 和 contact 结论分开报告；
- 明确说明该验证只覆盖 dataset 分布，不保证任意长时间 bitwise identity。

只有当 held-out RL distribution 上的单步 qacc、短时 rollout、obs/reward 分布都通过，才可以声称：

```text
OpenSim model + explicit runtime dynamics adapter is validated as a MuJoCo counterpart on the tested RL distribution.
```

不能声称：

```text
OpenSim native .osim is identical to MuJoCo.
```

也不能声称：

```text
The two simulators will produce exactly identical arbitrary long rollouts.
```
