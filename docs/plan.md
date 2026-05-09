你是一个熟悉 MuJoCo、OpenSim、肌骨建模、动力学一致性验证、数值诊断和工程修复流程的高级工程师。

现在的任务不是重新创建一套验证框架，而是在当前仓库已有的 `src/msk_equivalence` 框架上继续补强验证能力，并针对当前 Level 4 失败问题进行定位、诊断、修复和重新验证。

========================
一、当前背景
========================

本仓库中，MuJoCo 全身肌骨模型是 reference implementation，OpenSim 模型是对 MuJoCo 模型的复刻版本 / calibrated counterpart / surrogate。

MuJoCo 参考模型：

MimicMSK_Model_mujoco/body/myofullbody.xml

OpenSim 待验证模型：

MimicMSK_Model_opensim/MimicMSK_OpenSim.osim

主 mapping/config：

configs/model_mapping.yaml

现有验证框架：

src/msk_equivalence/

当前主验证命令：

PYTHONPATH=src MSK_EQUIVALENCE_SKIP_PLOTS=1 /data3/yangfeiyang/conda_envs/mujoco-opensim/bin/python -m msk_equivalence.compare \
  --osim MimicMSK_Model_opensim/MimicMSK_OpenSim.osim \
  --mjcf MimicMSK_Model_mujoco/body/myofullbody.xml \
  --mapping configs/model_mapping.yaml \
  --out results/equivalence_report \
  --fail-on-gate

当前状态：

Level 0: passed
Level 1: passed
Level 2: passed
Level 3: passed
Level 4: failed
Verdict: not equivalent

这意味着：

OpenSim 模型目前已经在结构、运动学、肌肉几何、惯性参数、静态/瞬时动力学上基本对齐 MuJoCo reference，但在 random activation qacc 和 short-horizon rollout 上仍然不等价，因此还不能称为 behaviorally equivalent，也不能称为 RL-interchangeable。

请严格遵守这个判断，不要在报告中夸大结论。

========================
二、最高优先级目标
========================

当前最重要的任务不是泛泛地补脚本，而是解释 Level 4 为什么失败，并建立从 Level 3 到 Level 4 之间的中间诊断层。

请优先完成：

1. random activation generalized torque decomposition；
2. per-coordinate / per-muscle contribution analysis；
3. random FK + orientation audit；
4. full inverse dynamics term checks；
5. passive joint grid / passive velocity sweep；
6. contact behavior checks；
7. repair loop / backup / before-after report；
8. provenance / model modification log。

不要创建新的 `validation_mujoco_opensim/` 目录，除非现有框架完全无法承载。默认应在：

src/msk_equivalence/checks/
src/msk_equivalence/report/
configs/dynamics_samples/
results/equivalence_report/

中扩展。

========================
三、需要保持的核心原则
========================

1. MuJoCo 模型是 reference，不允许修改。
2. OpenSim 模型可以修，但必须保守修复。
3. 不允许通过放宽阈值、隐藏失败坐标、删除失败项来制造通过。
4. 如果某些 finger/toe/distal coordinates 不属于 RL actionable coordinates，可以单独创建 RL-subspace report，但 full-body strict report 必须保留。
5. 不要只看 qacc error。必须把 qacc error 分解为：
   - muscle force mismatch；
   - moment arm mismatch；
   - generalized torque mismatch；
   - mass matrix / inertia amplification；
   - coordinate sign / coupling mismatch；
   - passive force / damping / limit force mismatch。
6. 所有新检查如果因为 API 限制无法运行，必须标记为 SKIPPED，不允许标记为 PASS。
7. 所有修复必须有：
   - backup；
   - git diff；
   - before / after metrics；
   - repair log；
   - affected-level revalidation；
   - 如果变差则回滚。
8. 所有 fitted correction force 必须用 held-out samples 验证，不能只在拟合样本上通过。
9. OpenSim `Thelen2003Muscle` 存在 minimum activation floor，因此 passive/zero-command 对比必须显式记录 OpenSim 和 MuJoCo activation/control 语义差异。

========================
四、新版验证层级
========================

请将验证层级更新为如下结构。

--------------------------------
Level 0: Structure / Topology / Conventions
--------------------------------

已有：
- topology
- conventions

需要补强：
1. unmapped dynamically-relevant coordinate audit；
2. joint axis audit；
3. joint range audit；
4. body / coordinate / muscle / actuator mapping completeness report；
5. root frame / gravity / unit convention report。

输出建议：
- results/equivalence_report/joint_axis_audit.csv
- results/equivalence_report/joint_range_audit.csv
- results/equivalence_report/unmapped_dynamic_items.csv
- results/equivalence_report/convention_audit.md

--------------------------------
Level 1: Kinematics
--------------------------------

已有：
- neutral kinematics
- joint sweep

需要补强：
1. random legal pose FK sampling，采样 500–1000 个合法姿态；
2. body/site/marker position error；
3. body orientation error；
4. per-body worst-case report；
5. multi-joint combination error report。

新增 check 建议：
- src/msk_equivalence/checks/random_fk.py
- src/msk_equivalence/checks/orientation_audit.py

输出建议：
- results/equivalence_report/random_fk_body_errors.csv
- results/equivalence_report/random_fk_site_errors.csv
- results/equivalence_report/random_fk_orientation_errors.csv
- results/equivalence_report/plots/random_fk_error_histogram.png
- results/equivalence_report/plots/random_fk_per_body_error.png
- results/equivalence_report/random_fk_report.md

要求：
- 单关节 sweep 通过并不代表多关节组合 FK 一定通过。
- 如果 random FK 失败，不要进入动力学修复，先修 mapping/sign/frame/joint axis。

--------------------------------
Level 2: Muscle Geometry
--------------------------------

已有：
- muscle_length
- moment_arm

需要补强：
1. constraint-chain-aware moment arm validation；
2. dependent coordinate moment arm report；
3. per-coordinate worst-muscle report；
4. failed Level 4 coordinates 的重点 moment arm 复查。

重点坐标：

mtp_angle_r
mtp_angle_l
md3_flexion_r
pm3_flexion_r
mcp3_flexion_r
mp_flexion_r
mp_flexion_l
ankle_angle_r
ankle_angle_l
subtalar_angle_r
subtalar_angle_l

新增输出：
- results/equivalence_report/moment_arm_dependent_coordinate_errors.csv
- results/equivalence_report/moment_arm_worst_by_coordinate.csv
- results/equivalence_report/moment_arm_priority_distal_report.md

要求：
- moment arm sign mismatch 是严重错误。
- 如果 moment arm 符号反，优先检查 coordinate sign、joint axis、path point、wrap object，不要先修 max_isometric_force。
- 如果 moment arm 局部跳变，检查 wrapping/via point。
- 如果 moment arm 幅值整体成比例偏差，检查 body scale、path point 坐标、单位。

--------------------------------
Level 3: Static / Instant Dynamics
--------------------------------

已有：
- inertial
- inverse_dynamics
- muscle_torque
- passive_forces

需要补强：
1. full inverse dynamics terms；
2. gravity term；
3. inertia term；
4. velocity-dependent term；
5. full random q/qdot/qddot dynamics；
6. passive force grid；
7. passive velocity sweep。

新增 check 建议：
- src/msk_equivalence/checks/inverse_dynamics_terms.py
- src/msk_equivalence/checks/passive_grid.py
- src/msk_equivalence/checks/passive_velocity_sweep.py

输出建议：
- results/equivalence_report/inverse_dynamics_gravity_term.csv
- results/equivalence_report/inverse_dynamics_inertia_term.csv
- results/equivalence_report/inverse_dynamics_velocity_term.csv
- results/equivalence_report/inverse_dynamics_full_random.csv
- results/equivalence_report/passive_joint_grid_errors.csv
- results/equivalence_report/passive_velocity_sweep_errors.csv
- results/equivalence_report/inverse_dynamics_terms_report.md
- results/equivalence_report/passive_grid_report.md

测试设计：
1. Gravity test:
   qdot = 0
   qddot = 0

2. Inertia test:
   qdot = 0
   qddot != 0

3. Velocity-dependent test:
   qdot != 0
   qddot = 0

4. Full random dynamics:
   qdot != 0
   qddot != 0

如果 gravity term 失败，优先检查：
- mass；
- COM；
- gravity direction；
- root frame；
- coordinate sign。

如果 inertia term 失败，优先检查：
- inertia tensor；
- body frame；
- joint axis；
- generalized coordinate order。

如果 velocity-dependent term 失败，优先检查：
- joint hierarchy；
- frame transform；
- coordinate coupling；
- inertia tensor orientation。

--------------------------------
Level 3.5: Generalized Torque Decomposition
--------------------------------

这是本次最重要的新增层级。

目的：

在进入 Level 4 qacc / rollout 之前，先比较 MuJoCo 和 OpenSim 在同一 q、qdot、activation/control 下产生的 generalized torque 是否一致。

因为：

qacc = M(q)^(-1) * (tau - bias)

如果 distal coordinate inertia 很小，即使 tau 误差不大，也可能被放大成巨大 qacc error。

因此必须把 qacc mismatch 拆开分析。

新增 check 建议：
- src/msk_equivalence/checks/generalized_torque.py
- src/msk_equivalence/checks/torque_decomposition.py
- src/msk_equivalence/checks/random_activation_torque.py

需要实现以下检查：

1. single-muscle generalized torque
   每次只激活一个 muscle / actuator，比较其对所有 coordinates 的 torque contribution。

2. group activation generalized torque
   按功能肌群激活，例如 ankle group、toe group、finger group、hip group、knee group、shoulder group。

3. sparse random activation generalized torque
   使用与 Level 4 random activation qacc 相同的 activation vectors。

4. per-coordinate torque reconstruction
   对每个坐标计算：

   tau_muscle = sum_i F_i * r_i

   其中：
   F_i 为 muscle/actuator force；
   r_i 为 moment arm。

5. inertia amplification analysis
   对 failing coordinate 报告：
   - generalized torque error；
   - diagonal mass/inertia scale；
   - qacc error；
   - qacc error / torque error amplification factor。

6. source ranking
   对每个 failure row 自动判断主要来源：
   - muscle force mismatch；
   - moment arm mismatch；
   - torque sign mismatch；
   - unexpected coupling；
   - passive/damping/limit force mismatch；
   - mass matrix amplification。

输出：
- results/equivalence_report/diagnostics/random_activation_torque_errors.csv
- results/equivalence_report/diagnostics/per_coordinate_torque_decomposition.csv
- results/equivalence_report/diagnostics/per_muscle_contribution_worst_rows.csv
- results/equivalence_report/diagnostics/inertia_amplification_report.csv
- results/equivalence_report/diagnostics/active_muscles_in_failing_vectors.csv
- results/equivalence_report/generalized_torque_report.md

必须支持 focused debug mode：

python -m msk_equivalence.compare \
  --checks generalized_torque \
  --debug-coordinate mtp_angle_r

python -m msk_equivalence.compare \
  --checks generalized_torque \
  --debug-vector-index 123

python -m msk_equivalence.compare \
  --checks generalized_torque \
  --debug-muscle <muscle_name>

要求：
- 如果 generalized torque 已经不一致，就先修 torque/moment-arm/force，不要直接修 qacc。
- 如果 generalized torque 基本一致但 qacc 不一致，重点检查 mass matrix、bias force、constraint/coupling、coordinate ordering。
- 如果 torque 误差很小但 qacc 很大，报告为 inertia amplification，而不是直接判定 muscle path 错误。

--------------------------------
Level 4: Behavior / Forward Dynamics
--------------------------------

已有：
- neutral no-contact qacc；
- sparse random activation qacc；
- 200 ms matched-activation no-contact rollout。

当前失败：
- random activation qacc 失败；
- 200 ms rollout 失败；
- 最坏坐标集中在 small-inertia distal coordinates。

需要补强：

1. 分阶段 rollout：
   - 10 ms smoke test；
   - 50 ms no-contact；
   - 100 ms no-contact；
   - 200 ms no-contact；
   - task/action-distribution rollout；
   - contact rollout。

2. qacc 失败诊断必须调用 Level 3.5 decomposition。

3. 支持 full-body strict mode 和 rl-actionable mode：

full_body_strict:
- 所有 mapped coordinates / muscles / actuators 参与统计。

rl_actionable:
- 只统计 RL 实际使用的 coordinates / actuators / state variables。
- 不允许隐藏 full_body_strict 结果。
- 如果某些 finger/toe coordinates 不参与 RL，必须在报告中单独说明。

输出：
- results/equivalence_report/forward_dynamics_10ms.csv
- results/equivalence_report/forward_dynamics_50ms.csv
- results/equivalence_report/forward_dynamics_100ms.csv
- results/equivalence_report/forward_dynamics_200ms.csv
- results/equivalence_report/forward_dynamics_task_distribution.csv
- results/equivalence_report/forward_dynamics_full_body_vs_rl_actionable.md

要求：
- 不要把 200 ms rollout 作为唯一标准。
- 先用 10/50 ms 定位短期动力学发散源。
- contact rollout 只能在 no-contact rollout 基本通过后再做。
- 不要要求长时间 bitwise identity，因为 OpenSim 和 MuJoCo 积分器、约束求解器、接触模型不同。

--------------------------------
Level 5: Contact Behavior
--------------------------------

当前 contact 检查更像 inventory / mapping，不足以证明接触行为等价。

需要新增 contact behavior checks。

新增 check 建议：
- src/msk_equivalence/checks/contact_behavior.py

检查内容：
1. contact geometry mapping；
2. contact onset timing；
3. normal force；
4. tangential/friction force；
5. impulse；
6. center of pressure，如果可获得；
7. foot slip；
8. foot penetration；
9. drop test；
10. stance probe；
11. simple foot contact rollout。

输出：
- results/equivalence_report/contact_force_errors.csv
- results/equivalence_report/contact_timing_errors.csv
- results/equivalence_report/contact_impulse_errors.csv
- results/equivalence_report/foot_slip_errors.csv
- results/equivalence_report/contact_behavior_report.md

要求：
- contact equivalence 是单独 claim。
- 即使 Level 4 no-contact 通过，也不能自动声称 contact equivalent。
- 如果 contact model 语义差异太大，报告中要明确写出限制。

--------------------------------
Level 6: Provenance / Standardness / Model History
--------------------------------

如果要证明 OpenSim 模型“可信”或“标准”，还需要 provenance 文档，而不是只看数值验证。

新增脚本建议：
- src/msk_equivalence/report/provenance.py

需要记录：

1. MuJoCo reference model path；
2. OpenSim model path；
3. git commit hash；
4. model file hash；
5. mapping file hash；
6. OpenSim model edits since initial import；
7. parameter diff；
8. correction forces；
9. manually tuned muscle parameters；
10. known deviations between OpenSim and MuJoCo；
11. Thelen2003Muscle activation floor handling；
12. validation command；
13. validation date；
14. environment information；
15. dependency versions。

输出：
- results/equivalence_report/model_provenance.md
- results/equivalence_report/model_parameter_diff.csv
- results/equivalence_report/model_file_hashes.json
- results/equivalence_report/known_model_deviations.md

要求：
- 不要声称这是 canonical standard human model。
- 正确说法是 calibrated OpenSim counterpart of the MuJoCo reference。
- 如果没有文献或实验验证，不要声称 biomechanically validated human model。

========================
五、自动修复流程
========================

请将 repair loop 工程化。

新增脚本建议：
- src/msk_equivalence/repair.py
- src/msk_equivalence/checks/repair_diagnostics.py

或在现有 compare.py 中增加保守的 --auto-repair 入口。

自动修复优先级：

Priority 1: mapping-level repair
- coordinate sign；
- coordinate offset；
- coordinate order；
- muscle/actuator mapping；
- body/site mapping。

Priority 2: frame / coordinate repair
- root frame；
- joint axis；
- parent/child frame；
- joint range；
- body local frame。

Priority 3: muscle geometry repair
- origin；
- insertion；
- via point；
- wrap object；
- path point body assignment。

Priority 4: muscle parameter repair
- tendon_slack_length；
- optimal_fiber_length；
- max_isometric_force；
- pennation angle。

Priority 5: fitted correction force
- only for clearly documented smoke-test correction；
- must be validated on held-out samples；
- must not be used to hide random activation or rollout failure。

禁止：
1. 修改 MuJoCo reference；
2. 提高阈值以通过；
3. 删除失败坐标以通过；
4. 只修结果、不管物理意义；
5. 在 FK 没过时修 muscle parameter；
6. 在 moment arm sign 错误时先修 max_isometric_force；
7. broad group tuning when failure is localized；
8. 不写日志就修改 .osim。

每次 repair iteration 必须：

1. 保存 OpenSim backup：
   results/equivalence_report/backups/MimicMSK_OpenSim_iterXX.osim

2. 写 repair log：
   results/equivalence_report/repair_log.md

3. 写 before/after metrics：
   results/equivalence_report/repair_before_after.csv

4. 运行 affected-level revalidation。

5. 如果 Level 0-3 退化或 Level 4 更差，则自动回滚。

6. 最多自动迭代 5 轮；超过后停止并给人工检查建议。

========================
六、优先实现顺序
========================

请不要一次性泛化重写所有模块。按以下优先级执行。

--------------------------------
P0: 直接解释 Level 4 失败
--------------------------------

实现：

1. generalized_torque.py
2. torque_decomposition.py
3. random_activation_torque.py

目标：

对 worst random activation qacc rows 进行逐坐标、逐肌肉、逐力矩贡献分解。

必须输出：
- failing vector index；
- failing coordinate；
- OpenSim qacc；
- MuJoCo qacc；
- active MuJoCo actuators；
- mapped OpenSim muscles；
- per-muscle force；
- per-muscle moment arm；
- per-muscle torque contribution；
- total torque mismatch；
- inertia amplification factor；
- likely failure source。

--------------------------------
P1: 补 random FK 和 orientation audit
--------------------------------

实现：

1. random_fk.py
2. orientation_audit.py
3. joint_axis_range_audit.py

目标：

确认不是多关节组合姿态或 frame orientation 隐藏问题导致 Level 4 爆炸。

--------------------------------
P2: 补完整 inverse dynamics terms
--------------------------------

实现：

1. inverse_dynamics_terms.py

目标：

把 inverse dynamics 分解为：
- gravity；
- inertia；
- velocity-dependent；
- full random dynamics。

--------------------------------
P3: 补 passive force grid 和 velocity sweep
--------------------------------

实现：

1. passive_grid.py
2. passive_velocity_sweep.py

目标：

验证 neutral pose 之外的 passive muscle / joint limit / damping behavior。

--------------------------------
P4: 补 contact behavior
--------------------------------

实现：

1. contact_behavior.py

目标：

从 contact inventory 升级为 contact force / timing / slip / impulse behavior validation。

--------------------------------
P5: 补 provenance 和 repair loop
--------------------------------

实现：

1. provenance.py
2. repair.py
3. repair_log generation

目标：

使模型修改过程可追踪、可回滚、可复现。

========================
七、命令行接口要求
========================

请扩展现有 compare.py，使其支持以下用法。

运行完整验证：

PYTHONPATH=src MSK_EQUIVALENCE_SKIP_PLOTS=1 /data3/yangfeiyang/conda_envs/mujoco-opensim/bin/python -m msk_equivalence.compare \
  --osim MimicMSK_Model_opensim/MimicMSK_OpenSim.osim \
  --mjcf MimicMSK_Model_mujoco/body/myofullbody.xml \
  --mapping configs/model_mapping.yaml \
  --out results/equivalence_report \
  --fail-on-gate

运行 Level 3.5 generalized torque：

PYTHONPATH=src MSK_EQUIVALENCE_SKIP_PLOTS=1 /data3/yangfeiyang/conda_envs/mujoco-opensim/bin/python -m msk_equivalence.compare \
  --osim MimicMSK_Model_opensim/MimicMSK_OpenSim.osim \
  --mjcf MimicMSK_Model_mujoco/body/myofullbody.xml \
  --mapping configs/model_mapping.yaml \
  --out results/equivalence_report \
  --checks generalized_torque torque_decomposition random_activation_torque \
  --fail-on-gate

运行 random FK：

PYTHONPATH=src MSK_EQUIVALENCE_SKIP_PLOTS=1 /data3/yangfeiyang/conda_envs/mujoco-opensim/bin/python -m msk_equivalence.compare \
  --osim MimicMSK_Model_opensim/MimicMSK_OpenSim.osim \
  --mjcf MimicMSK_Model_mujoco/body/myofullbody.xml \
  --mapping configs/model_mapping.yaml \
  --out results/equivalence_report \
  --checks random_fk orientation_audit \
  --fail-on-gate

运行完整 inverse dynamics terms：

PYTHONPATH=src MSK_EQUIVALENCE_SKIP_PLOTS=1 /data3/yangfeiyang/conda_envs/mujoco-opensim/bin/python -m msk_equivalence.compare \
  --osim MimicMSK_Model_opensim/MimicMSK_OpenSim.osim \
  --mjcf MimicMSK_Model_mujoco/body/myofullbody.xml \
  --mapping configs/model_mapping.yaml \
  --out results/equivalence_report \
  --checks inverse_dynamics_terms \
  --fail-on-gate

运行 Level 4 focused debug：

PYTHONPATH=src MSK_EQUIVALENCE_SKIP_PLOTS=1 /data3/yangfeiyang/conda_envs/mujoco-opensim/bin/python -m msk_equivalence.compare \
  --osim MimicMSK_Model_opensim/MimicMSK_OpenSim.osim \
  --mjcf MimicMSK_Model_mujoco/body/myofullbody.xml \
  --mapping configs/model_mapping.yaml \
  --out results/equivalence_report \
  --checks forward_dynamics generalized_torque torque_decomposition \
  --debug-coordinate mtp_angle_r \
  --fail-on-gate

检查 Python：

/data3/yangfeiyang/conda_envs/mujoco-opensim/bin/python -m compileall src/msk_equivalence

检查 diff：

git diff --check

========================
八、报告要求
========================

请更新现有 report system，使最终报告明确分层声明当前模型可以声称什么、不能声称什么。

报告必须包含：

1. Level 0-6 status；
2. full-body strict report；
3. RL-actionable report，如果 mapping 中定义了 RL subspace；
4. key metrics；
5. worst failure rows；
6. per-coordinate failure ranking；
7. per-muscle contribution ranking；
8. qacc mismatch decomposition；
9. torque mismatch vs inertia amplification；
10. passive grid result；
11. contact behavior result；
12. model provenance；
13. repair history；
14. known limitations；
15. final claim。

最终 claim 必须按照证据分级：

- structurally equivalent；
- kinematically equivalent；
- muscle-geometry equivalent；
- static-dynamically equivalent；
- behaviorally equivalent；
- RL-interchangeable；
- contact-equivalent。

当前在 Level 4 失败时，正确结论应类似：

The OpenSim model is validated through Level 3 against the MuJoCo reference, including structure, kinematics, muscle geometry, inertial properties, and static/instant dynamics. However, strict Level 4 random-activation qacc and short-horizon rollout tests still fail, especially on small-inertia distal coordinates. Therefore, the model should not yet be treated as behaviorally equivalent or RL-interchangeable with the MuJoCo reference.

========================
九、Definition of Done
========================

本阶段完成标准：

1. 不新建平行验证框架，而是在 `src/msk_equivalence` 内扩展。
2. 新增 random FK + orientation audit。
3. 新增 full inverse dynamics term checks。
4. 新增 Level 3.5 generalized torque decomposition。
5. 新增 passive joint grid / passive velocity sweep。
6. 新增 contact behavior check，至少形成基础版本。
7. 新增 provenance report。
8. 新增 repair log / backup / before-after report。
9. 能够解释当前 Level 4 random activation qacc failure 的主要来源。
10. 重新运行相关 checks。
11. 报告明确说明哪些层级通过、哪些层级失败。
12. 不夸大结论，不把 Level 0-3 通过说成 RL 等价。

最终强目标：

OpenSim 模型只有在以下条件满足时，才可以称为 validated MuJoCo counterpart：

1. Level 0-3 仍然通过；
2. Level 3.5 generalized torque decomposition 通过；
3. Level 4 random activation qacc 通过；
4. Level 4 no-contact short rollout 通过；
5. RL action/state distribution rollout 通过；
6. contact rollout 如果被声称，则必须单独通过；
7. diagnostics 中没有隐藏的 coordinate/muscle group extreme error；
8. 报告明确说明测试分布、限制和不保证任意长时间 bitwise identity。

请现在开始执行：

1. 阅读当前 repo；
2. 检查 `src/msk_equivalence` 现有结构；
3. 不重建框架；
4. 优先实现 P0 generalized torque decomposition；
5. 然后实现 P1 random FK + orientation audit；
6. 运行 compileall；
7. 运行相关 validation；
8. 生成报告；
9. 给出失败定位和下一步修复建议。