# go2 楼梯步态适配（路线 A：本体感知盲爬）— 设计与验证计划

**[2026-09-24 状态：暂停，步态修改已回退]**。Phase 1 平地回归 B 组两次翻倒，
根因已定位（见文末"暂停时的根因"），恢复前先按"修正方向"改反射判定。
WIP 存档：`sim/robots/go2/stair_adapt_wip/`（含适配版 go2_controller.py、
fk_check.py、routeA.patch）；恢复 = 把前两个文件拷回 `sim/robots/go2/`。
当前仓库里的 go2_controller.py 是 HEAD 原版（已验证平地行为正常）。

2026-09-24 首轮实现，改动脉冲全部加性、可门控，平地回归是第一验收项。

## 1. 根因与思路

sim 的 go2 是 IK treadmill trot：落足 z = 体相对固定值（`swing_height +
robot_height`），控制器假设地面高度恒定。上台阶时前摆足撞踢面、支撑面每升
一级误差累积——实测 8cm 踢面最多上一级，15cm 完全上不去。

路线 A = 给步态补"支撑面在哪"的本体感知，不改步态结构：

1. **正解 FK**：订阅 `joint_states`（encoder 实际值），`InverseKinematics.
   foot_position_body()` 反解每条腿足端在体系下的位置。FK 是 IK 的代数精确
   逆（fk_check T1：140 样本最大误差 1.4e-16 m）。
2. **支撑面估计**：stance 足 z 中位数 = `ground_ref`（体系，负值）；前后对
   支撑差 = 爬升信号。
3. **落足高度闭环**：摆动目标加 `swing_z_bias`——平地=前腿小探查
   （0.02m），反射升级后=估计的踢面高度（步进 0.02、上限 0.18，覆盖 15cm
   踢面+余量）。发现踢面的机制：**卡足反射**——摆动腿指令目标比 FK 实际高
   2cm 以上持续 5 tick = 足被踢面挡住 → 共享偏置升级。
4. **俯仰**：`pitch_cmd = 1.2 × (前支撑 − 后支撑)`，限幅 ±10°、速率 5°/s、
   死区 1.5cm。实现走**体系落足目标旋转**（`roty(pitch_cmd)`，上游 Pupper
   的 IMU 补偿同款路径）；`body_local_orientation` 走不通——hip 变换链把
   z 轴重映射了，俯仰会等量落到前后腿（实测 -0.024/-0.026，fk_check 注释有
   记录）。
5. **限速**：爬升中 vx 钳到 0.02 m/s，退出带 100 tick 滞回。
6. **偏置衰减**：3 秒无证据后每 10 tick 衰减 0.01（~2.4s 忘光，防平地跺脚）。

**回归安全设计**：所有修正是加性项，零地形误差时数学上恒等于原行为（swing
偏置 0 时代码逐位等价，fk_check T3 1e-15 验证）；`stair_adapt:=false`（默认
值）整个估计器旁路，与 master 行为一致。

## 2. 改动清单

- `sim/robots/go2/go2_controller.py`：FK 方法、`SupportEstimator`、State 加
  `swing_z_bias`/`pitch_cmd`、swing 目标加偏置项、step() 加俯仰旋转、节点加
  参数/joint_states 订阅/主循环接线、`STAIR_*` 常数表。
- `sim/robots/go2/fk_check.py`：离线单元检查（stub ROS，容器直接跑，无需 sim）。
- `sim/worlds/{stairs.sdf,gen_stairs_sdf.py}`：楼梯世界（踢面 0.15/踏面
  0.28/每段 8 级，L 折返，转本生成器重出）。

## 3. 验证计划（未执行）

### Phase 0 — 离线单元（已完成 2026-09-24，全绿）
`python3 sim/robots/go2/fk_check.py`：T1 FK 往返 1.4e-16；T2 俯仰差动方向；
T3 零偏置=legacy（1e-15）；T4 平地语义；T5 合成楼梯反射/俯仰/爬升标志；
T6 衰减。

### Phase 1 — 平地回归（第一优先，空世界）
- 启动：`bash sim/run_simulator.sh --world sim/worlds/empty.sdf`（cpp 栈或
  sensor 栈），go2 出生原点。
- 驱动脚本分五段：静止踏步 10s → 前进 0.04×20s → 停 5s → 前进+偏航
  (0.04, wz 0.3)×15s → 横移 0.01×10s；全程 10Hz 采 gz 真值 + gt_twist 落盘。
- 三个对照组：`stair_adapt:=false`（新代码旁路）、master 原版、
  `stair_adapt:=true`。
- 通过判据：
  - false vs master：轨迹差 < 2cm、速度差 < 10%（改动旁路证明）；
  - true 平地：|v−v_des| ≤ 0.008 m/s；躯干 z 波动 ≤ 1.5cm；偏航速率差
    < 10%；全程 climbing=False、step_bias=0（无假触发）；不摔倒。

### Phase 2 — 楼梯爬升（stairs.sdf，15cm 标准踢面）
- P2.1 助跑+第一段 8 级：到达一级平台（z 1.2±0.05）；中途卡死 >10s 判失败；
  记录每级耗时与 pitch 曲线（应贴梯面上翘、到平台回零）。
- P2.2 平台转向 yaw +90°：不出护栏。
- P2.3 第二段到顶（z 2.4±0.05）。
- 全程录 rosbag（/cmd_vel、gt_twist、joint_states、gz pose）复盘。
- 预期失败模式→对策：踩沿打滑→升 mu/降 probe 速度；IK 饱和抖振→降
  pitch 限幅；反射振荡→加大升级滞回；下楼不在本阶段（踩空探测另做）。

### Phase 3 — 导航链路（等建图 session 空闲）
`/mapping/start` → 驱动上楼 → `/mapping/stop` → 校验 path_climb 标签（梯段
climbing=1、平地=0）→ `--map --map-dir` 回灌 → 发目标跨楼梯导航 →
dog_state 验收。速度先验应自动在梯段限速（建图时开得慢）。

### 回滚
`stair_adapt:=false` 即恢复 legacy；git 层面 go2_controller.py 单文件独立回退。

## 4. 暂停时的根因（2026-09-24，Phase 1 B 组实测）

平地 A(off)/B(on) 对比：A 正常（1.17m/43s，z 波动 2.5cm）；B 翻倒
（z 波动 39cm、终值 +0.33）。两处已实证的缺陷：

1. **joint_states 乱序（已修）**：broadcaster 按**注册哈希序**发关节
   （实测 `lf_lower, rf_hip, lf_hip, lh_upper, ...`），按位置映射全读错。
   修复 = 按 joint 名字查表（`SupportEstimator.on_joint_states`）；修后静止
   FK 读数 −0.250 精确、climbing=0。恢复时必须带 T7（乱序映射用例，用实测
   顺序构造）。
2. **卡足反射被正常摆动滞后误触发（主因，未修）**：正常摆动中指令 z 走三角
   波、物理足有惯性滞后，"指令比实际高 2cm 持续 5 tick"在**平地每次摆动都
   成立** → step_bias 一路升到 0.18 上限（debug 实测 mean 0.10、climbing=1
   占 57%、front_sup 被自己的高踩偏置从 −0.25 抬到 −0.07 的假象）→ 前足越
   踩越高 → 平地自翻。
   **修正方向**：阻塞签名改 XY 平面（踢面挡的是前进方向，水平跟踪误差
   >3-4cm 持续 4 tick；正常摆动水平滞后仅 1-2cm，无 z 波形歧义），或
   "误差递增且足速≈0"；再加摆动相位门（只在落足段 prop>0.6 判定）。
3. **限速单位 bug（次因，未修）**：STAIR_SPEED_GATE=0.02 是 plant 单位
   （×13 增益 ≈ 0.26 m/s 实际）。改相对门控（爬升时 vx×0.5）。
4. 附带：pitch 死区 1.5cm 偏紧（trot 躯干俯仰噪声 ~1cm），恢复时放宽到
   2.5cm + 持续 10 tick。

fk_check 恢复时同步加：T8 正常摆动滞后不触发反射（合成案例：xy 滞后 1.5cm
+ z 滞后 5cm，断言 bias 不变）。
