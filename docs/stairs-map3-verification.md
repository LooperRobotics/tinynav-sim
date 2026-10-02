# map3 楼梯策略 RL 训练与验收协议

适用范围：`go2-map3-stairs` 训练线（MotrixLab `stairs_exp/`）及其 gsplat
部署集成。本文档是训练 stage 的开工契约：**门槛在 stage 开始前定稿，训练
中途不改；每个"验收通过"的说法必须附带本文档规定的证据产物；最终判决权
在人眼**（看 Layer 2/3 的视频），agent 的职责是生成证据和标出可疑项。

## 0. 为什么需要这份协议

历史教训：

- **单数字指标会说谎**。站立吸引子的 reward 比走路高（实测 1.7/step）；
  双稳态转弯的 `gz_mean` 可以看起来正常——冻住的 episode 和猛转的
  episode 一平均就"达标"了。v12 训练里转弯正常、部署直接冻住（接触通道
  语义不匹配），训练 env 的 eval 完全没暴露。
- **2500-iter checkpoint 是截断的**。最终 run（`26-10-01_21-05-27`）末尾
  reward 仍在涨（1764→1884→1927，无平台），验收时所有"软/弱/双稳态"
  症状第一嫌疑人是 undertraining。

因此验收信号必须是：**分能力记分卡（抓分布级造假）+ 确定性视频（人眼
判决）+ 部署面复测（抓训练/部署差异）+ 回归门（防修 A 坏 B）**。

## 1. 四层验收协议

### Layer 1 — 能力记分卡（每 checkpoint 自动）

工具：`stairs_exp/eval_scorecard.py`（数据协议继承 `eval_map3.py`，
29 spawn、seed 11）。

```bash
cd stairs_exp/MotrixLab
# 出记分卡（不改任何文件，exit code = 判决）
.venv/bin/python ../eval_scorecard.py <run_dir或ckpt.pt> \
    --stage stage1_turn --tag s1_iter800
# 与已验收基线对比（回归检查，见 Layer 4）
.venv/bin/python ../eval_scorecard.py <ckpt> --stage stage1_turn \
    --compare <baseline_scorecard.json>
```

指标与门槛全部在 `stairs_exp/verify_gates.json`（**stage 开工前定稿**）。
核心防造假设计：

| 能力 | 指标 | 抓什么假好 |
|---|---|---|
| 转向 | 按 \|wz\|∈{0.2,0.4,0.6} 分桶 err + **freeze 比例**（单 episode 均值 \|gz\|<0.25\|cmd\| 判冻结） | 双稳态：均值正常但一半 episode 冻住 |
| 前进/后退 | 分桶 vx err + falls | 只会在一个速度段走 |
| 指令切换 | 直行→转弯响应延迟（\|gz\|>0.25 首达时间） | 指令恒定训练出的"没见过切换" |
| 楼梯 | 爬楼/下楼计数（Δfloor>0.5 m） | 用摔下去凑位移 |
| 步态形态 | 对角/同侧接触相关、步频、**四足摆动顶点散布**、躯干高度/倾角 | 兔子蹦（同侧相关为正）、前探腿（单脚 apex 异常高）、匍匐（躯干高度低） |
| 全部 | falls per scenario | 靠 episode 重置刷距离 |

### Layer 2 — 确定性视频（人眼判决的主力）

工具：`stairs_exp/record_scenarios.py`。固定剧本（stand / fwd / fwd_stop /
turn_L / turn_R / reverse / stairs_up / stairs_down）、固定 spawn、固定
种子、固定指令序列，每个 checkpoint 产出可逐帧对比的 mp4。

```bash
.venv/bin/python ../record_scenarios.py <ckpt> --out /tmp/videos --tag s1 \
    --vs <baseline_ckpt>     # 额外产出 cmp_*.mp4：左基线右候选
```

规则：

- **stage 验收 = 记分卡 PASS + 用户看过该 stage 的 montage**（总时长
  2–3 min）。没有视频和记分卡两样产物，不允许说"好了"。
- 训练过程可视化：`snapshot_training_videos.py <run_dir> --every 500`
  按 iter 间隔录 shorts，用户能看到步态从哪一刻开始成型/漂移，而不是
  只看终点数字。

### Layer 3 — 部署面验收（抓训练/部署差异）

工具：`gsplat/tools/deploy_acceptance.py`（容器内运行）。真实部署链
（onnx → 桥 → cmd 文件 → sensor_server → 物理）走脚本化 /cmd_vel，
录 color 流 mp4 + GT 轨迹。

```bash
# 容器内，sim 已起（run_gsplat.sh，supervisor on）
/opt/venv/bin/python gsplat/tools/deploy_acceptance.py --out /tmp/acc_flat --scenario flat
/opt/venv/bin/python gsplat/tools/deploy_acceptance.py --out /tmp/acc_stairs --scenario stairs
# stairs 场景需先把狗放到上行梯段出生点（pick_spawn.py）
```

每个 stage 至少做 flat 精简版；`release` stage 必须 flat+stairs 全做。
v12 教训：训练 env 全绿、部署冻住——只有这一层能暴露。

### Layer 4 — 回归门

`--compare <baseline>` 强制全量对比：除 stage 要求的指标外，**记分卡上
所有与基线可比的项目不得回退超过 `regress` 容差**（verify_gates.json
每项配了绝对回退量）。任一格子掉了 → checkpoint 拒绝，无论新能力涨多少。

基线管理：每个被接受的 stage 产出正式记分卡
（`scorecard_<tag>.json`），下一个 stage 以它为 compare 基线；
`release` 以 2500-iter 现状记分卡为基线
（`runs/go2-map3-stairs/rslrl/torch/ppo/26-10-01_21-05-27-425915/scorecard_base2500.json`，
已实测：release 门槛 28 项中 17 项 FAIL——转向 freeze 36–48%、爬楼 1/29、
直行 err 0.19–0.72，与人工实测症状一致，可作回归对比的锚点）。

## 2. 训练 stage 计划

| Stage | 内容 | 训练改动 | 验收（Layer 1 门槛见 gates） |
|---|---|---|---|
| 0 续命 | 纯续训当前 ckpt 500–800 iter | 无（同一 cfg resume） | 不设门槛；记分卡对比 2500-iter 基线，确认"只是没训够" |
| 1 转向 | 出生点限平台/平地，指令 turn-heavy（纯转 40%，\|wz\| 0.1–0.6 连续覆盖），**episode 中途 resample 指令** | `go2_map3.py` 训练 cfg | `turn_*`、`trans_*` 全过 + 全量回归 + montage 人看 |
| 2 后退+直行 | vel_limit 拓 [-0.6,1.0]，后退 slow band | cfg | `rev_*`、`lin_*` + 回归 + montage |
| 3 爬楼 | 上行出生点对称加密，foothold_gain 上调 | cfg + spawn 表 | `trav_*` + 回归 + montage |

每 stage 训练中途按 `--every 500` 出 snapshot 视频（Layer 2b）。

### 中途验证节奏（stage 内）

1. 训练跑到 1/2 预算时：先跑 Layer 1 记分卡（informational）+ 一条
   fwd snapshot 视频。若形态在往坏方向漂（蹦跳/前探/匍匐），**提前终止
   并复盘奖励**，不烧完预算。
2. 训练结束：Layer 1（--stage）→ Layer 2（--vs 基线）→ 用户看 montage
   → Layer 3 flat → 回归对比。全过才标记 stage 完成。
3. 任一步 FAIL：记录进实验文档的"尝试"一节（改了什么、判决结果），
   下一个变体从上一个**已验收** checkpoint 出发，不从失败变体继续。

## 3. 集成前置（RL 之后的仿真修复队列）

部署验收有意义的前提是仿真面健康，待修（详见 `gsplat-sim-progress.md`
与实验文档）：

1. **IMU 实际 100 Hz**（楼梯 dt=0.01 时 `sensor_server.py` 的 IMU
   累加器每步只补一拍）——改 `if` 为 `while`。
2. **摔倒检测盲区**：趴伏（根高 0.19–0.22 m）与怠速下蹲（0.218 m）
   重叠，单一高度阈值无解 → 指令-运动死锁判据（非零指令 + 低位 +
   2 s 位移 < ε → 复位）。
3. **viewer 1 Hz 劣化**（运行中途掉进不可恢复的 wgpu 呈现路径）→
   Hz 看门狗自动重启 viewer。
4. teleop 后退限幅 -0.3（训练包络同步扩展前的速赢）。

## 4. 产物索引

| 产物 | 位置 |
|---|---|
| 门槛定义 | `stairs_exp/verify_gates.json` |
| Layer 1 记分卡 | `stairs_exp/eval_scorecard.py` |
| Layer 2 录制/对比 | `stairs_exp/record_scenarios.py` |
| Layer 2b 训练快照 | `stairs_exp/snapshot_training_videos.py` |
| Layer 3 部署验收 | `gsplat/tools/deploy_acceptance.py` |
| 记分卡存档 | `<run_dir>/scorecard_<tag>.json` |
| stage 记录 | `docs/stairs-map3-experiments.md`（按节追加） |
