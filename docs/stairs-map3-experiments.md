# go2 楼梯策略实验总结与待解决问题（map3 楼梯间）

覆盖从 v12 转弯优化到 map3 楼梯间训练、部署、实测反馈的完整实验线。
训练/验收协议（四层验收 + stage 门槛）见 `docs/stairs-map3-verification.md`。
早期版本细节见 `stairs_exp/爬楼训练问题与尝试总结.md`（v1–v13 逐次记录）；
步态与奖励设计另见 `docs/stairs-gait.md`。

> **当前状态**：Stage 0-3 finetune 链的指标最优产物 s2
> 被用户视觉判决**否决**（步态退化、右后腿持续高抬），全链训练停止；
> 基线重新成为部署候选。详见「用户视觉判决」一节。

## 目标

go2 在 map3 五层楼梯间（3DGS 视觉 + 123 逐构件碰撞盒）中：遥控行走、
上/下楼梯、摔倒自动恢复；远期接导航闭环。

## 实验过程

### Stage 0 续训（验收协议首用）

- 方案：2500-iter ckpt 纯续训 800 iter（同分布同权重，resume 至 3300）。
- 假设：reward 仍在爬（末三点 1764→1884→1927）→ "只是没训够"。
- **结果：假设被记分卡否决。** reward 涨到 2271 但 turn freeze 全面恶化
  （0.2/0.4/0.6 桶：48%→64% / 38%→53% / 36%→45%），turn err 同步变差，
  lin falls 5→7——更多训练在同一分布下**加深了欠转平衡**（与奖励经济
  分析一致：tracking_ang 0.5 / σ0.15 时欠转成本 ≈0.009/step，步态稳定
  收入轻易覆盖）。`scorecard_s0.json` 已存（对比基线 `scorecard_base2500.json`）。
- 结论：分布/权重必须动 → 进 Stage 1（s1turn：平地出生点、turn-heavy
  全频段、episode 中途 resample、ang_vel 权重 0.5→1.0），从 s0 ckpt
  出发（更高的行走收入与 episode 长度是资产；转向恶化正是 Stage 1
  要改的目标）。
- 流程笔记：`eval_scorecard.py --compare` 的方向性回归检查有效， Stage 0
  无门槛 PASS 只表示"跑完了"——判决看数字对比，不看 VERDICT 行。

### Stage 1 转向专项（第一轮 700 iter）

- 改动（`go2-map3-stairs-s1turn`）：平地 5 出生点、40% 全频段纯转
  （|wz| 0.1–0.6）+ 30% 带速转弯、**episode 中途 resample 指令**（2.5–5 s，
  训练分布首次包含指令切换）、tracking_ang_vel 权重 0.5→1.0。
- 结果（`scorecard_s1.json`，对比 s0）：方向正确但未达门槛。
  turn freeze 0.4/0.6 桶 53%→29%、45%→29%（减半）；0.2 桶 64%→55%；
  trav_descends 7→8、trav/gait falls 4/4→2/1；**步态形态代价**：
  diag_FL_RR +0.098→−0.005、pair_RL_RR −0.51→+0.08（trot 相位松动）、
  cadence 1.4→1.9 Hz。turn 绝对门槛（freeze ≤20–25%、err ≤0.12–0.15）
  全 FAIL → VERDICT FAIL。
- 解读：双稳态吸引子很深，700 iter 只挖掉一半；奖励曲线仍在爬。
  决策：**同 cfg 再续 700 iter**（方向已验证，收益递减前再压一轮）；
  若 0.4/0.6 桶 freeze 仍 >40%，升级结构改动（指令历史进 obs / 相位钟）
  作为 Stage 1c。门槛严格度（尤其 0.2 小桶）留给用户在 release 前的
  montage 判决时一起定。

### Stage 1 第二轮与 1c 分岔

- 第二轮（同 cfg 续 700 iter，`scorecard_s1r2.json`）：**波动无净进步**
  ——0.2 桶 freeze 55%→36% 但 0.6 桶 29%→45%，0.4 桶 err 回升。分布
  杠杆已花完，同 cfg 再加 iter 只会震荡。
- 按预案升级 Stage 1c：不用未验证的结构改动，改用 **v 线已判决过的
  sharp-yaw 配方**——tracking_ang_sigma 0.15→0.10（v12 docstring：
  σ0.15 时 0.448/0.5 欠转是平衡点而非优化噪声，σ0.10 让近零误差梯度
  陡 3 倍），分布保持 s1turn，600 iter，从 r1 ckpt 出发。
- **s1c 结果（`scorecard_s1c.json`）：σ0.10 无效甚至更差**（0.6 桶
  freeze 60%，历史最差）。事后归因：v12 的 σ 杠杆治的是"转得动但欠转"，
  本 lineage 的问题是"冻结/双稳态"——冻结 episode 在任何 σ 下 tracking
  收入都 ≈0，梯度陡缓无关。**Stage 1 定型：接受 r1（01-40-03）为链上
  ckpt**（0.4/0.6 桶 freeze 29%/29% 为全线最好），0.2 小桶与绝对 err
  门槛未达的事实带入 release 由用户判决。结构改动（相位钟）留待
  release 后单独立项，不再阻塞链。

### Stage 2 后退扩展（700 iter）

- 改动（`go2-map3-stairs-s2rev`）：vel_limit 拓 [-0.6,1.0]、20% 后退
  slow band（含 -0.1~-0.3 轻退）、25% 纯转保留 Stage 1 质量、中途
  resample 继承。
- 结果（`scorecard_s2.json`，对比 s1r1）：**后退训练有效**——
  rev_0.4_err 0.378→0.227、rev_0.2_err 0.195→0.127；意外收益：
  turn_0.2 freeze 55%→**17%**（首次过 20% 绝对门槛）；全量回归 PASS。
  绝对 err 门槛（lin/rev ≤0.10-0.15）仍未达——这些是基线就FAIL的
  严格线，release 时连同 montage 交用户判决。
- 链上 ckpt 更新为 s2（03-15-11）。

### Stage 3 爬楼两轮与最终定型

- r1（900 iter，`scorecard_s3.json`）：climbs 1→3 但 descends 8→5、
  falls 2→5 双回归 → 同 cfg 第二轮。
- r2（900 iter，`scorecard_s3r2.json` + release 复测）：falls/descends
  收复（3/7），但 climbs 回落 1，且 **turn 大幅回退**（0.4 桶 freeze
  36%→64%、0.2 桶 17%→33%）——1800 iter 的爬楼专项没有换来稳定的爬楼
  能力，反而伤了转向。**map3 真楼梯爬升不是 foothold_gain+出生点质量
  能解决的**，需要 v 线 v9→v10 那样的多轮专属迭代，本次预算内不再追。
- **最终候选定型为 s2**（`runs/go2-map3-stairs-s2rev/.../26-10-02_03-15-11-123206`）：
  四项用户目标里转向（0.2 桶 freeze 17% 历史最佳）、后退（rev_0.4
  0.378→0.227）、直行保持全线最优，爬楼维持基线水平（1/30）。
  ONNX 已导出（parity 6.9e-6）部署为 `go2_stairs_map3.onnx`。
- 记分卡噪声标定：同 ckpt 两次跑计数类指标有 ±1-2 抖动（混沌翻转
  单个 episode 后 RNG 级联），freeze 比例级差异才是信号。

### Release 验收（s2 最终候选）

- 记分卡（`verify_artifacts/release_s2_scorecard.log`，对比 2500-iter
  基线）：**无回归；turn freeze 48%→21% / 38%→28% / 36%→24%**（0.6 桶
  首次过绝对门槛 ≤25%）；rev_0.4 0.241→0.217、rev_0.2 0.112→0.119；
  descends 12→11。**VERDICT 仍 FAIL**——卡在训练前定下的严格绝对线：
  lin/rev err ≤0.10-0.15（基线即 FAIL，全线从未达到）、trav_climbs ≥10
  （最好 3/30）、cadence ≥2 Hz（当前 1.1-1.9）。
- 产物：`verify_artifacts/s2final_vs_base/`（8 场景同屏 montage，左基线
  右 s2，供人眼判决）；`verify_artifacts/s1_vs_s0/`。
- 部署面验收（容器，s2 onnx 已部署为 `go2_stairs_map3.onnx`，parity
  6.9e-6）：flat 场景 24.5s 位移 0.99 m、净转向 -31.8°（左右转指令都
  有响应）、零跌落；stairs 场景 25.5s 位移 1.14 m、零跌落、F4 层内
  行走但**未爬升**（与训练 eval climbs 1/30 一致，爬楼能力未获）。
  产物容器内 `/tmp/accept/{flat,stairs}/{color.mp4,gt_trace.json}`。
- ~~待用户判决~~：**已有判决（见下节）——s2 被否决，全链训练停止。**

### 用户视觉判决：s2 否决，训练停止

- 用户人眼检查 montage 后判决：**s2 步态比 2500-iter 基线更差，不可
  接受**；最明显的问题是右后腿持续高抬（非步态过渡相位）。
- agent 抽帧复核（`verify_artifacts/s2final_vs_base/gait_reject_frames/`，
  左基线右 s2 同屏）证实：
  - **stand 场景**：s2 侧右后腿在 clip 10%/30%/50%/70% 采样点全部保持
    高抬，爪端离地约一个躯干高；基线同场景四脚落地。
  - **fwd 场景**：s2 侧两条后腿同时向后上方甩（膝关节深弯、爪端高于
    脚踝），呈"蹬地后不收腿"的蹦跳形态；基线后腿支撑相正常。
  - **turn_L 场景**：s2 侧有一腿跨过体中线高抬；基线四肢均低位。
- 训练已全部停止（host 无 train.py / 录制进程，GPU 0%）。
- **核心教训：记分卡指标盲区**。release 记分卡 s2 全线"无回归"、freeze
  大幅改善，但 freeze/err/cadence 全是运动学结果指标，没有任何一项
  约束"站立时四脚着地/关节姿态不离谱"——**记分卡 PASS 不能替代视觉
  验收**。这正是"很多次 zcode 说好了、人眼一看有新问题"的机制。
- 此后每轮 finetune 验收协议必须加关节级/视觉级门（量化版"步态正常"）：
  1. 站立/零速场景**四脚接触率**门（足端高度阈值 + 接触比例下限）；
  2. **单腿持续悬空时长**上限（如 >0.5 s 记 fail）；
  3. 髋/膝关节角与默认站姿偏差上限（|Δq| 逐关节阈值）；
  4. montage 必须先由 agent 抽帧自查关节姿态、再交用户人眼双签。
- 下一步候选方向（等用户定）：
  - a. 从基线重新 finetune 转向/后退，但把**形态保持项**（站立接触
    penalty + 关节偏差 penalty）写进 reward，每轮过关节级门；
  - b. 部署回滚基线 onnx，转向/后退用上层面绕过（指令塑形/状态机）；
  - c. 加大基础训练预算从头训，形态保持进 base reward。
- 部署侧注意：容器内 `go2_stairs_map3.onnx` 当前是 s2 版，判决后应
  回滚基线版（基线 ckpt `runs/go2-map3-stairs/.../26-10-01_21-05-27-425915`）。

### 仿真修复实测（容器 tinynav）

| 修复 | 验证 | 结果 |
|---|---|---|
| IMU 100→200 Hz | sensor_server 30s 冒烟 + 全 rig 桥报 | **200.0 Hz**（修复前 100；桥侧 216–221 Hz） |
| 摔倒死锁判据 | 容器内三项单测 | 趴伏+指令 **2.0 s 触发**；站立+指令 / 趴伏+零指令均不误触发 |
| viewer 看门狗 | `--gui` rig 冒烟 | 检测/计数/重启循环工作正常；但本次 docker exec 起 tmux 的上下文里 viewer **从启动即 1 Hz**（wgpu 呈现路径老问题，重启也不恢复）——看门狗机制达标，根因在启动环境，已记录待查 |
| teleop 后退限幅 | 发 vx=-0.5 查 cmd 文件 | **-0.4000**（钳到训练包络，wz 不受影响） |

观察项（非阻塞）：桥侧 IMU 计数比 server 产量持续多 ~10%（10-01 与
10-02 两拨日志同现），ring `read_imu` 单调用无重复路径，疑为 reader
侧簿记或 ring 重初始化边界，VIO 实际速率已达标，登记待查。

### v12：sharp-yaw 转弯跟踪（scale 1.0 / σ0.10）

- 部署侧小 wz 精度优秀；持续大 wz spin 会衰减停转。
- 注入因果实验定位根因：**接触通道语义不匹配** —— 训练读 link 合力
  （切向为主、方向多变），部署 pair 传感器读法向（恒 ≈[0,0,1]）。

### v13：接触通道风格随机化（p=0.5）——失败废弃

- 两个 checkpoint 全部塌成"短暂转→站定"。机理：50% 质量压在低信息
  语义上，PPO 收敛到"忽略接触通道+保守站立"吸引子。
- 教训：**语义级随机化 ≠ 噪声增广**，不能这样跨域。
  代码保留（`contact_proxy_prob` 默认 0）。

### map3 仿真环境搭建

- 底层地板闪烁 = 与 F1 平面共面 z-fighting → catch plane 下沉 5 cm。
- "踏上楼梯掉到一楼"排查：踏步本身密实（probe 逐点投放验证），
  真实孔洞是中央楼梯井（61 cm，贯穿五层）与东缝（24 cm）→
  加红色半透护栏（alpha 0.1）：井两侧、缝两侧、四面外圈墙，
  probe 验证 9/9 推挤 + 6/6 投放不穿不漏。
- 出生点 F4(-6, 4) 朝西（用户实驾后指定）。

### v12 接入 gsplat 部署链

- 新 rig `go2_sensor_rig_stairs.xml`（496 pair 接触传感器、
  12 电机 ±24、FL/FR/RL/RR 序）。
- `sensor_server.py` 新增 `Go2StairsLocomotionPolicy`：
  60 维 obs（含 12 维接触通道 = 首个命中 pair 的 found 法向转到体系）、
  100 Hz 软件 PD 力矩。配置 `gsplat/configs/map3_go2.json`。

### dt 覆盖坑（部署步态崩掉的根因）

- MJCF 合并规则：scene 的 `<option>` 覆盖机器人 option；scene 只写了
  gravity → timestep 被重置为引擎默认 → 策略步态崩（"往前走就跌倒"）。
- 训练后端把 dt 强制为 SimCfg=0.01 且每 ctrl 步仅 1 个物理子步；
  部署 scene 必须钉 `timestep="0.01" integrator="Euler" iterations="60"`。
  （0.005×2 子步会让转向动力学反号；引擎版本无关——0.10.1 与
  0.7.1.dev 同 XML 位级一致，已复验排除。）

### 按用户方案转向：躯干碰撞 + 楼梯环境内训练

- **躯干碰撞**：原 MJCF 只有 11.4×9.35×11.4 cm 核心小盒（狗会"穿"
  踏步立面）；对齐 gz go2 的 37.62×18.7×11.4 cm 全身盒，
  训练/部署/键盘 demo 三处同步。
- **训练环境** `go2_map3.py`（`Go2Map3StairsTask`）：
  - 场景孪生 `scene_go2_map3.xml`（同一 map3_collision.xml + 护栏）。
  - AABB 地面查询替代 hfield 采样（多层楼：ceiling = root_z − 0.05
    选层规则），逐点 Python 循环 2.8 s/iter 向量化后降到 2.37 s/iter。
  - 29 出生点（五层/平台/踏步中段/下行口，带朝向四元数与楼层提示）。
  - 指令分布：25% 纯转 |wz|∈[0.3,0.6]、20% 慢速带；
    vel_limit vx∈[-0.4,1.0]、wz∈[-0.6,0.6]。
  - `gait_diagonal` 步态塑形（对角同步 − 同侧同步，EMA+ramp）。
  - 摔倒终止：z < spawn_gnd − 2.5（掉井兜底）+ 原有倾覆判定。

### 训练过程坑（2500 iter 从头训练）

- configclass 模糊 setattr：先赋值后声明字段会静默覆写近似名字字段
  （`tracking_ang_sigma` 写进 `tracking_sigma`），冒烟打印才发现。
- **站立吸引子**：sharp-yaw 面板在 wz=0 时白送 1.0/步，新网络直接
  停在静止最优（站立收入 1.7/步）→ 转弯面板回退 0.5/σ0.15。
- kill 打中 bash 包装 PID 而非 python 本体 → 三个训练并发日志串线。
- 传感器 query 平铺布局 (N,1984) 而非 (N,496,4)；注册顺序 envcfg 先于
  env；广播 (123,) vs (8,) ceiling 形状对齐。

### 验收结果（29 出生点，`stairs_exp/eval_map3.py`）

| 项目 | 结果 |
| --- | --- |
| TROT 步态 | 达成（diag 相位 +0.30/+0.20，pair −0.53） |
| 下楼 | 12/29 出生点走完 |
| 爬楼 | 2/29（弱） |
| 域内转弯 | ±0.21–0.35 rad/s（双稳态） |
| 原地后退 | 0.38 m/s（域内） |
| 部署前进 | 0.46 m/s ✓ |
| 直行跟踪 | 0.17–0.25（软） |

### 部署侧修复（最近一轮实测后）

1. **viewer 光源**：顶光（`dir="0 0 -1"`）使下层被上层楼板遮暗 →
   改为**侧面无限远平行光**（`dir="-1 -0.35 -0.65"`，E-NE 约 32° 俯角）。
   渲染探针三变体对比（纯水平 47 / 32° 斜侧 78.5 / 45° 南向 81.6 均值
   亮度 vs 顶光 95.6），选 32° 斜侧写入 `map3_scene/mjcf/scene.xml`。
   探针方法：`RenderApp` + `get_camera(0).capture()`，**等待期间必须
   持续 `render.sync` 泵帧**否则拿空帧；主机侧 X11 抓屏对 rootless
   Xwayland 无效（全黑），只能走 capture API。
2. **出生姿态**：出生 z 12.308 → 12.173（落高 0.45 → 0.31 m）+
   出生/复位后 0.5 s **PD hold 阶段**（不走 ONNX，直接 PD 到默认角，
   last_action 保持 0）→ 消除落地瞬态甩腿（用户所见"一只脚在后"）。
3. **摔倒监督 `StairsSupervisor`**（`sensor_server.py`）：
   - 判据：躯干倾覆（z·up < 0.3）**或** 距局部 AABB 地面高度 < 0.18 m
     持续 1 s → 复位回配置出生点（`data.reset` 会回 MJCF 原点，改为
     重放配置 qpos + FK + hold）。
   - 环境旋钮 `TINYNAV_GS_FALL_HEIGHT`。
   - 验证：阈值 0.25（出生站高 0.277 之下不触发）零误复位；
     阈值 0.30 强制触发 → 1.1 s 周期复位循环、位置正确回出生点。

## 待解决问题（按优先级）

1. **s2 已否决（用户判决）**：转向/后退专项 finetune 换来的
   指标改善伴随步态形态退化（右后腿持续高抬等），见上节。基线重新成为
   部署候选。下轮训练必须带形态保持 reward + 关节级验收门。
   部署侧待办：容器内 onnx 回滚基线版（s2 版覆盖前留过 .bak 的可恢复，
   否则从基线 ckpt 重导）。
2. **验收协议缺关节级门（本次事故根因）**：eval_scorecard 只有运动学
   结果指标。TODO：record_scenarios/eval_scorecard 增加足端接触率、
   单腿悬空时长、|Δq| 偏差三项自动判定，先 agent 抽帧自查再交用户。

3. **摔倒检测盲区（用户实测复现：整轮 0 次复位）**
   实测"摔倒"形态 = 行走后趴伏（根高 0.19–0.22 m）与正常怠速下蹲
   （0.218 m）高度重叠，单一高度阈值无法区分（实测狗趴在 0.188，
   距阈值仅 8 mm）。候选：
   - 指令-运动死锁判据：有非零指令 + 高度持续偏低 + 位置不动 > 2 s
     → 复位（区分"趴着不动"与"蹲着待命"最可靠的信号是给指令后动不动）；
   - 高度 + 关节 sprawl 联合判据；
   - 根治：训练自恢复（get-up）行为——改动大，需单独立项。
4. **原地转弯双稳态**：同一 wz 指令按步态相位要么冻住要么 0.32 rad/s
   猛转。候选：更长训练 / 纯转指令占比提高 / 相位钟塑形 / 指令历史进 obs。
   （s2 曾把 freeze 压到 17-24%，但已被否决——下轮必须连带形态保持项
   一起训，不能单独追这个指标。）
5. **后退控制**：teleop ↓ 发 vx=-0.5，超出训练包络 [-0.4, 1.0] →
   行为未定义。速赢：teleop 后退限幅已做（-0.4）；根治：下轮训练把
   vel_limit 拓到 [-0.6, 1.0]（s2 已验证能学，同样要带形态保持项）。
6. **光照仍不满意（用户反馈）**：侧面平行光已五层均匀照亮但整体亮度
   下降。候选：侧光 + 弱顶光双灯、headlight ambient 0.3→0.5、
   再抬仰角。需要用户指认具体不满意点（哪一侧/哪一层/整体太暗）。
7. **爬楼弱（2/29，s3 专项 1-3/30 也未解）**：foothold_gain 通路已在
   但训练质量偏下楼；Stage 3 已证明单轮 foothold_gain+出生点加密不够，
   需 v9→v10 式多轮专项，单独立项。
8. **直行速度跟踪软**（域内 0.17–0.25；s2 的 rev 改善随否决作废）。

## 产物清单

- **PIE 视觉策略迁移线**：wukong 机器训练的深度相机楼梯策略
  （policy_12998.onnx）将作为第二条策略线接入 gs-playground（转弯/上下楼
  能力覆盖 v12 短板，无接触通道失配问题）。迁移差距与训练侧待办
  （后退/原地转/摔倒恢复/深度相机对齐）见
  `~/workspace/github/RL/stairs/go2_pie_20261002/docs/go2-pie-gsplat-migration.md`。
- 训练（stairs_exp / MotrixLab）：
  `motrix_envs/src/motrix_envs/locomotion/quadruped/go2_map3.py`、
  `.../go1/xmls/scene_go2_map3.xml`、`configs/task/go2-map3-stairs/`、
  checkpoint `runs/go2-map3-stairs/.../26-10-01_21-05-27-425915`、
  验收脚本 `stairs_exp/eval_map3.py`
- 判决证据（s2 否决）：
  `verify_artifacts/s2final_vs_base/gait_reject_frames/`（同屏抽帧，
  左基线右 s2）、`verify_artifacts/release_s2_scorecard.log`、
  被否 ckpt `runs/go2-map3-stairs-s2rev/.../26-10-02_03-15-11-123206`
  （仅作对照保留，勿部署）
- 部署（tinynav-sim / gs_playground）：
  `gsplat/configs/map3_go2.json`、`gsplat/server/sensor_server.py`
  （`Go2StairsLocomotionPolicy` + `StairsSupervisor`）、
  `map3_scene/mjcf/scene.xml`（光源/护栏/dt 钉死）、
  `models/robots/navigation/go2/go2_sensor_rig_stairs.xml`、
  `policies/go2_stairs_map3.onnx`（**当前为被否的 s2 版，待回滚基线**）

## PIE 策略线部署更新（后退/原地转战役收官）

九轮训练战报与三个判决性发现见 wukong 机器
`~/workspace/github/RL/docs/go2-backward-inplace-campaign.md`（本地归档
`~/workspace/github/RL/stairs/go2_pie_20261002/` 同步）。部署侧变更：

- `mujoco/assets/policy/policy.onnx` 换为 **model_18997**（楼梯 lv5 上13/下14、
  原地转 16/16 @0.3rad/s 精确跟踪、前进 16/16；sha256 ff63892e）。
- `policy_12998_backup.onnx` 旧部署版备份（sha f4dab830）——已移出仓内，
  需要时从 RL 训练仓 model_12998 重导。
- `mujoco/sim/keyboard.py`：`WZ_HOLD` 0.5→0.3（狗头 10cm/s 上限 ⇒ wz≤0.33）；
  后退键 −0.5→−0.2（新常量 `VX_HOLD_BWD`，对齐平地后退训练带）。
- **双专家切换器已落地**（`mujoco/sim/expert.py`）：`view.py` 默认双专家
  `--expert auto`——后退命令（vx<−0.05 迟滞 + 50 tick 驻留）自动切平地专家，
  其余走楼梯主力。切换时目标专家
  GRU 清零（episode-fresh 即训练分布自身的复位语义，跨网拷贝隐状态无意义），
  观测历史共享不盲窗。`--expert single --policy <file>` 退回单策略。
  切换器验证：分段指令自动往返 flat↔stairs、驻留无抖动、全程直立零摔。
- **注意**：默认形态下 18997 承担前进/楼梯，↓ 键自动触发平地专家——后退
  直接可用，无需手动切换。
- 旧 gsplat 线 v12/map3 产物清单（上文）未动，独立并存。

### 统一策略战役：不可达，平地专家升级换血

单模型"楼梯+真后退"课程（U1 纯平地后退原型 → U2 楼梯回归保鲜）在 wukong
机器跑完并判决**不可达**：U1 的 model_24000 后退原型成型且质量超专才
（−0.2 16/16@94%、四带速度分级），但 6000 纯平地迭代把楼梯技能清零
（上下楼 0/16）；U2 楼梯 1000 迭代闪电恢复到上13/下15（18997 深度先验残迹
红利）后继续退化，同一窗口内后退原型被清零、原地转也被冲刷——两个技能块在
~1000 迭代尺度互相擦除，"保鲜剂量"无效。战报与三教训见
`~/workspace/github/RL/docs/go2-unified-backward-campaign.md`（本地归档
`~/workspace/github/RL/stairs/go2_pie_20261002/docs/` 同步）。

部署侧净收益（战报岔路 A）：

- **平地全向专家升级**：`policy_10496_flat_omni.onnx`（后退 11/16@72%，已删，
  sha cc77739f）→ **`policy_24000_flat_omni.onnx`**（=U1 model_24000，
  sha256 fc6b1a79）。训练探针：后退 −0.2 16/16@94%、四带分级
  −0.05→1.07 / −0.1→1.82 / −0.15→2.86 / −0.2→3.76 m、前进 16/16、原地转
  16/16。
- 本仓容器 smoke 复现（`smoke.py flat --vx −0.2 --onnx …`，`--onnx` 为
  本次新增的策略覆盖参数）：−0.2×10s 位移 −1.89m（跟踪 94.5%、upright 1.0、
  未摔）；0.5×5s +2.32m（93%）。onnx 图契约与现役 policy.onnx 逐输入输出
  一致（proprio 45 / proprio_history 450 / depth_history 2×60×86 /
  memory 128 → actions 12），元数据契约（joint/order/action_scale）同源。
- **默认策略不变**：`policy.onnx`（model_18997）仍是部署默认——统一候选点
  楼梯 0/16 或续训不稳，均不可替换默认。
