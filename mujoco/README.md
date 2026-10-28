# MuJoCo dog sim: 策略驱动 + wgpu 3DGS 传感器视图 + HIL 双传输面

Unitree 四足（默认 Go2，见机型注册表）由自训的深度策略（ONNX）驱动，在 map3
楼梯井里行走（MuJoCo 物理），
狗眼相机画面由 gausscam（原 splatsense）的 wgpu 管线渲染（3DGS）。整体跑在
tinynav-sim 仓，作为"整体仿真"侧：既可以人机交互遛狗，也可以作为传感器源喂
navcore 做验证。

## 文件架构

```
mujoco/
├── README.md            本文件
├── view.py              交互宿主：viser 网页 viewer（默认，含操作面板、运行中
│                        切地图 / 切机型）；--preview 才建 splat 管线并开 MJPEG 推流
├── webpanel.py          网页操作面板（view/hil 共用）：地图 / 机型 / 感知 / 手柄 /
│                        跟随视角 / 防撞墙 / 加速度 / Reset·Stop / 策略深度图 /
│                        胯关节温度
├── physics_view.py      纯物理 viewer 宿主（无 splat）——姿态/物理排查
├── smoke.py             headless 验证：flat | map3 | down（--robot 选机型）
├── hil.py               HIL 宿主入口：rclpy 契约面直发 DDS（无桥、单传输）
├── sim/                 仿真核心包（与传输/显示无关）
│   ├── contract.py      策略 I/O 契约（vendor 自上游 sim2sim，常量勿改）
│   ├── contract_v2.py   深度学生策略契约族（按 ONNX 元数据：足式 47 维、
│                        轮足 55 维，腿位置 + 轮速度动作）
│   ├── robots.py        机型注册表：MJCF / 网格 / 相机安装位 / 感知·盲策略
│   ├── policy.py        ONNX 策略会话（GRU memory + 元数据契约校验，CPU 单线程）
│   ├── plant.py         世界搭建：场景 + 注册表机型 + 执行器内 PD + rig 相机
│   ├── runtime.py       50Hz 控制循环（深度 10Hz → obs → 策略 → 4×mj_step）
│   ├── keyboard.py      /dev/input 全局按键轮询（零依赖）
│   ├── gamepad.py       /dev/input 手柄（evdev 事件流，零依赖）
│   ├── accel.py         键盘 / 手柄指令的最大加速度（斜坡限幅）
│   ├── guard.py         防撞墙：深度图看到正前方的墙就不给前进速度
│   ├── thermal.py       胯关节电机温度估算（关节力矩 → 铜损 → 绕组 / 外壳两节点）
│   └── tests/           纯逻辑单测（契约族、手柄、加速度、防撞墙；无需设备）
├── hil/                 HIL 发布面（Face 协议 + rclpy 实现，宿主循环共享）
│   ├── __init__.py      Face 协议 + 传感器契约常量（话题/QoS/TF/内参）+ 采样槽
│   └── ros.py           rclpy 发布面（节点名 /insight_full；rclpy 只在此导入）
├── viser_client/        vendored viser 网页客户端源码（v1.1.1，已剥键盘相机
│                        键位；构建见其 README，镜像 node 阶段自动编译）
├── webviewer/           headless 网页 viewer（vendored mjviser/viser，
│                        hil.py --view web 用；无 DISPLAY 依赖）
├── assets/
│   ├── mjcf/            map3 scene.xml + map3_collision.xml（123 box，唯一源）
│   ├── .compose/        plant.py 运行时生成的预处理场景（gitignore，勿手改）
│   ├── go2/             go2_pie.xml（Go2 机体骨架；网格在 model/ 资产包，见下）
│   ├── <robot>/         go2w/as2/as2w/b2/b2w 训练 MJCF（训练仓导出，含执行器与
│                        碰撞配置，勿手改；网格在 model/<robot>/assets）
│   └── policy/          <robot>_depth_v*.onnx（各机型深度学生，机型注册表用）
│                        + policy.sha256（训练侧校验清单）
└── scripts/
    ├── launch.sh        统一 tmux 启动器（sim | mj | hil）
    ├── decimate_mesh.py 机型视觉网格减面 → MSH（model/ 资产包用）
    └── probe_viewer.py  X/GLFW 健康探测
```

资产来源：**不入 git 的大资产统一走 `model/` 资产包**——把 `model.zip`
解压到仓根即得（`model/splat/` = 场景，`model/go2/` = 减面网格；包内自带
`VERSION.txt` 与 `MANIFEST.sha256`，解压后 `sha256sum -c MANIFEST.sha256`
验包）。现在的机体要 2026-10-22 版的包（VERSION.txt 写明版本，2026-10-19 版的网格对不上）；
已有旧包时可只换机体网格：删掉 model/ 下 go2 go2w as2 as2w b2 b2w 六个目录，再解压
只含这六个目录与 VERSION.txt、MANIFEST.sha256 的小包，splat 不变。
重打包：`bash mujoco/scripts/pack_model.sh`（自动重生成
MANIFEST.sha256 再 zip，默认输出到仓外 `../model.zip`）。运行期查找链：
环境变量 `ROBOT_ASSETS` → 仓内 `model/` → 旧
`~/workspace/dm/robot-assets` checkout；mjsim 镜像把整个包烤在镜像内
仓根（与宿主机解压后布局一致），构建前放好 `model/` 即可。包内容出处：
`map3_scene.ply`
导出自 gsplat 侧 map3 建图 3DGS 场景（`examples/map3_stairs/assets_local/
map3_scene`，INRIA 布局，`from_ply` 直读 RAW 值）；`map2_scene.ply` 为
MetaCam map_2 的 spirula 训练输出直拷（step-50000，2.98M 高斯，706MB）；
`go2/` 为 robot-assets 仓的
减面二进制 MSH（1.3MB，出处链/重生成方法见该仓 README——该仓本地
checkout 尚无提交，网格版本以 MANIFEST 校验和为准）；`go2w/ as2/ as2w/ b2/
b2w/` 为各机型减面 MSH（训练仓 unitree_rl_mjlab 网格 →
`mujoco/scripts/decimate_mesh.py`，每个目录有 README 记出处）。
历史：本仓 `assets/go2/assets/` 曾带 25.5MB menagerie OBJ，
减面迁移后删除（上游 menagerie 与 splatsense examples 各有全量副本）。
入库的策略 ONNX 来自 RL 训练仓 unitree_rl_mjlab（hash 见 policy.sha256，
其 .pt 血缘在训练侧，本仓只带 onnx）。PIE 策略（policy.onnx = model_18997、
policy_24000_flat_omni.onnx）与双专家选择器已于 2026-10 移除，统一用自训
策略；实验档案见 ../docs/stairs-map3-experiments.md 与 git 历史。

## 命名约定

- 文件名不带策略/产品专名（策略论文名、对端盒子的产品名都不进文件名）；
  语义用通用词：contract/policy/plant/runtime/face/transport。
- `sim/contract.py`（策略契约，vendor 冻结）与 `hil/__init__.py`
  （传感器话题契约，对齐 looper 盒子 oracle）是两份不同语义的契约，分开放。
- `hil/ros.py` 是唯一允许 `import rclpy` 的地方。hil/ 包其余部分保持
  py3.10 兼容（可在不带 ROS 的 python 里直测契约与采样槽）。

## 三种形态

| 形态 | 入口 | 内容 | 用途 |
|---|---|---|---|
| hil（默认） | `hil.py` | DDS 契约面 + viser 网页 viewer（`--view glfw` 换原生窗） | 喂 navcore |
| sim | `view.py` | viser 网页 viewer + 操作面板（切地图/切机型/感知/手柄/跟随相机/深度图）+ 狗眼 splat（`--preview` 才建并推流），无 DDS | 人看/遥操 |
| mj | `physics_view.py` | 纯 viewer + 策略 | 物理排查 |
| dog | `dog.py`（或 `bash docker/run-mujoco-sim.sh dog -- --profile as2`） | 底盘 emu：go2 身体 + 真机 Unitree-DDS 脸（rt/* 话题 + sport RPC + 机型 FSM 门控），状态回传默认 20Hz | 多机型 pilot 驱动联调 |

emu 的工具链（全部入库，可重放）：
- **L0 单测**（纯逻辑免 SDK）：`python3 -m unittest robot_emu.tests.test_fsm robot_emu.tests.test_profile`（在 mujoco/ 下）。
- **conformance**：`python3 -m robot_emu.conformance --profile as2 --iface docker0`——对运行中的 emu 枚举 DCPSPublication，按审计 curated 清单逐 (topic,type) 核对（go2/as2 均 PASS）。
- **探针**：`python3 -m robot_emu.probe_as2|probe_go2 --iface docker0`——真 SDK 客户端核对线面事实 + FSM 门控序列，退出码即结论。
- **e2e**：emu 侧跑 `dog.py` 后，pilot 侧容器里 `ROBOT_TYPE=as2 python3 -m robot_emu.e2e_driver`（生产 unitree_control 全链路：拒动→stand→走一米→sit，含 /unitree/odometry 断言）与 `python3 -m robot_emu.e2e_charge`（charge_watch 充放电判定）。
- **场景注入**：JSON 发 `rt/emu/scenario`：`{"bms_ma":1500,"bms_soc":95}`（在桩充电）、`{"error_code":N}`、`{"refuse_move":true}`（障碍拒动，Move 回 -1 且机体停）、`{"cmd_blackout_s":2}`（断流 N 秒）、`{"reset":true}`（回出生点）、`{"clear":true}`。
- **默认场景 flat**（纯平地无坠落面，长跑测试用）；`--scene map2|map3` 复用 hil 场景。

dog/emu 的 DDS 依赖 unitree_sdk2py（纯 python；镜像 pip 安装 `mujoco/robot_emu/vendor/unitree_sdk2py` 里的 pin 源码检出——上游 `unitree_sdk2_python@814556d`，与 pilot 的 pyproject pin 同一 commit，子模块语义、提交时转正；树内直跑 DDS 模块前需先 `pip install` 该检出，L0 逻辑测试免安装）+ cyclonedds python 绑定。**版本必须 ≤0.10.x**：本机实测 cyclonedds 11 的 type object 会让 pilot 侧 0.10.2 解析段错误；mjsim (py3.12) 用 `CYCLONEDDS_HOME=<symlink farm→/opt/ros/jazzy>` 源码编 0.10.2 即可（其 libddsc 依赖 libiceoryx，也要链进 farm；Dockerfile 已把整套烤入）。**loopback 不能跑发现**（lo 无 IFF_MULTICAST，SDK 模板又没有 Peers 段）：bench 绑 `docker0`（有 MULTICAST flag、与真狗 123 网段隔离）。pilot 侧 ROS 2 与 SDK DDS 同域 0 并存无冲突；要严格隔离可在 bench 启动脚手架设 ROS_DOMAIN_ID（可选）。

可视化四通道：viser 网页 viewer（hil/sim 都默认；无 DISPLAY 依赖、CPU 约为
GLFW viewer 的 1/17，带浏览器 Reset/Stop 按钮——hil 里 Reset/Stop 只有按钮，
无键盘绑定）；MuJoCo 原生窗口（`--view glfw` 回退项，需 X）；浏览器 MJPEG
推流（`--preview` 显式开启，sim/hil 都是 :8888，第二屏看狗眼用）；rqt/ROS
工具直接订阅契约话题。
OpenCV/Qt 窗口形态已废除（混合图形机器上 Qt 是四客户端
段错误的成员之一；详细档案不入库，本机存于
`~/workspace/dm/archive/gausscam-archive/p2-wgpu-report.md` §7/§8）。

场景：`plant.SCENES` 注册表，hil/sim/mj 都吃 `--scene map3|map2|flat`（默认
map3；sim 的网页面板还能运行中切换）。**map3** = 楼梯井（123 box 精确碰撞，出生 F1 正对 L0 梯段）。**map2** = MetaCam map_2 室内（spirula 50k，2.98M 高斯；
平地 box + 边界墙碰撞，出生 `low` = gsplat 验证过的低采段点 (3.21,−5.54)，
yaw −11.5°——狗眼 0.45m 只有低采段渲染干净，**换出生点先 `--preview` 渲两
帧对质再走**）。策略跟机型走、与场景无关。碰撞地面高
度在 `map2_scene.xml` + 注册表各有一处，改要一起改；新场景 = 一份 mjcf +
注册表一个条目。

**[OPEN-1 →RESOLVED] 容器内 MuJoCo viewer "冻结"（已结案：不是冻结，是缺 sync）**

- **症状（回顾）**：mjsim 容器内 `mujoco.viewer.launch_passive` 的窗口只画
  一帧后"永久冻结"；物理、wgpu splat、DDS 数据面不受影响。
- **真根因（探针定案）**：**hil.py 主循环从未调用 `viewer.sync()`**。
  `launch_passive` 的语义就是"循环调 sync() 才更新"——viewer 渲染的是
  viewer 侧 mjData 副本，没有 sync 就永远显示启动瞬间的静态快照，viewer
  线程本身健康（空闲等待）。与容器/显卡/驱动**全部无关**，宿主上 hil 的
  viewer 同样是静态的（view.py/physics_view.py 都有 sync 所以正常，唯独
  hil.py 漏了）。当初"图像动但狗不动""只画一帧"全部由此而来。
- **旧结论作废**：先前"同进程 wgpu 的 NVIDIA Vulkan 初始化杀死 GLX 重绘
  循环"系误判——xwd 像素取证把"无 sync 的静态图"当成了"线程冻结"，死亡
  时点与 wgpu init（启动耗时 3-5s 的大头）重合纯属时间线巧合。
- **证伪链（scripts/probe_open1{,b,c}.py，容器内实测）**：viewer 先开 +
  帧循环带 sync（~150kHz）→ 依次叠加 adapter/device 初始化、完整 splat
  Pipeline、首帧 wgpu render、EGL Renderer（mujoco.Renderer，即 policy
  深度相机同款）→ **全程无一冻结**（12-14s 每秒 ~80k-150k 帧存活）。
  静态对照组 hil 进程线程画像：无任何 GL 繁忙线程（ Healthy idle）。
- **修复（已落地验证）**：hil.py 主循环补节流 `viewer.sync()`（30Hz 上限，
  物理迭代间检查）。容器内实测：狗走 R2b 楼梯时窗口画面连续变化（xwd 连续
  6 采样哈希全不同），wgpu+EGL+rclpy 同进程无任何冲突；cam 10Hz 零丢帧。
  **容器内 viewer 从此默认可用**——交互（转视角/Space/R）+ 实时遛狗观察
  不再依赖宿主侧进程/链路。
- **性能定则（sync 修复的直接代价）**：sync 每次触发一次
  viewer 场景渲染（render-per-sync 实证），**CPU 与 --view-hz 线性**：
  30Hz≈+87% 单核、10Hz≈+30-40%，默认 `--view-hz 10`，可按机器调。
  `__GL_SYNC_TO_VBLANK=0` 保留——vsync 开启会让 sync 阻塞在 swap 上
  （给 200Hz 控制环引入 16ms 抖动）。键盘移动驱狗（↑↓←→）是 hil/sim 全局
  捕获（/dev/input，hil 里键盘覆盖 DDS /cmd_vel）。
- **OpenBLAS 池自旋（第二根因，870% 元凶）**：numpy 自带的
  OpenBLAS 会在每次 BLAS 调用后整池自旋空等——PIE tick 200Hz 里的微型
  numpy 调用让 5-8 个线程持续烧核（10.5ms-cpu/tick，狗静止也烧）。镜像
  已钉 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`（10.5→0.9ms/tick）。
  同时钉 `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`（jazzy 默认 Fast DDS，
  其发现/日志线程额外烧核，且与 tinynav 栈的 CycloneDDS 不一致）。
  修复后（viewer@10Hz 活着）整机 ~85-90%，无 viewer ~50%。
- **远端 2060M/595 的真冲突（用户实测修复，与 OPEN-1 是两回事）**：
  在 Turing/595 驱动上 GLX viewer 先于 wgpu 初始化会让
  wgpuInstanceEnumerateAdapters panic（khronos-egl unwrap），且
  GLX+wgpu(Vulkan)+EGL 三者同进程无法共存（depth EGL makeCurrent 失败）。
  hil.py 已改 **wgpu-first 顺序 + viewer 强制 EGL**（glfw
  CONTEXT_CREATION_API hint）——在 5070/610 上顺序自由，此改动无副作用。
- **保留的教训**：① xwd 像素取证判"冻结"前必须先排除"无 sync 的静态图"；
  GL 窗口被遮挡时 Mesa Present 会跳过绘制（假冻结）；X11 窗口 ID 会复用
  （按 WM_CLASS 过滤 + 逐窗口置顶消歧）。② 环境变量组合（GLX vendor/
  PRIME/DRI3/vsync 等 7 组）全是安慰剂——症状在代码层，别在驱动层找。
  ③ 探针脚本模式（分阶段初始化 + 帧循环存活计数）值得复用。
  ④ 驱动代际差异真实存在：同一份代码 595/Turing 与 610/Blackwell 行为
  不同，跨机器结论要分开记录（本文件 [OPEN-1] vs 远端 2060M 段）。
  ⑤ 进程取证细节：pgrep 会匹配到自己，先遍历 /proc 按 comm/cmdline 校验；
  并发开第二个 GLFW viewer 会段错误，调试已有容器进程用
  `gdb -p $(docker-inspect pid)` 这类隔离入口，别在宿主再开 GL 客户端。
  ⑥ 编辑大文件后 grep 旧标记名查重复残留——hil.py 曾因重复 ctrl 分支导致
  快照永不写入，症状 = cam 0Hz 但 imu 200Hz 正常。
- **显示通路事实（顺手查清，回应"是否与 Intel 核显有关"——答案：无关）**：
  本机 = Ubuntu 24.04 GNOME Wayland，Intel ARL 核显（card1/renderD128）
  + RTX 5070 Laptop（card0/renderD129），PRIME `on-demand`。宿主显示与
  两侧 viewer 的 GL 都落在 **Intel**：gnome-shell 合成在 card1、Xwayland
  `:1` 的 provider 是 renderD128、宿主 GLX 默认渲染器 = Mesa Intel(R)
  Graphics (ARL)。**容器不是缺 Intel 卡**——run-mujoco-sim.sh 已透传全部
  4 个 DRI 节点，viewer GL 走 Mesa iris；容器内 Vulkan 侧只有注入的
  NVIDIA ICD。宿主/容器两侧同组合实测都健康。驱动为 610.43.02（旧文档
  写 595 是与老 2060 机器记串，已全部更正）。

## 机型、策略与网页操作面板

机型注册表 `sim/robots.py`：go2、go2w、as2、as2w、b2、b2w，机体都是 go2-blind 的
机器人模型：MJCF 是训练仓（unitree_rl_mjlab 分支 go2-depth）用
`scripts/go2_depth/export_sim_robot.py` 导出的训练原样机体（执行器 PD /
力矩上限 / armature、碰撞配置、轮子球形碰撞都在 XML 里，勿手改，改了重导；
旧的 PIE 机体 go2_pie.xml 已不在注册表里）；视觉网格是同一次导出的网格经
`scripts/decimate_mesh.py` 减面后的 MSH（open3d quadric 保 15%），放 model/ 资产包
`model/<robot>/assets`。深度相机安装位取训练值——**只有 Go2 是实测外参，其余机型是
按 Go2 方式放在头前的估计值**。场景与出生点通用，出生高度按机型
（`spawn_drop`）。

策略按机型登记：`perception`（深度学生，默认）与 `blind`（盲策略；没有盲
策略的机型感知开关锁定为开）。盲策略是 unitree_rl_mjlab go2-blind 分支的导出
（单个 `obs` 输入 = 10 帧逐项历史，MLP，无深度；指令包络 vx −1.0~1.5、vy ±0.6、
wz ±1.5），按 I/O 签名识别。关节目标与真机部署一致地截在默认角 ±2.0 rad 内
（训练时没有截断，正常运行几乎碰不到）。ONNX 元数据 `contract` 选契约
（`sim/contract_v2.py`）：go2_depth_v2 与 {as2,b2}_depth_v1 为 47 维，
go2w_depth_v1 为 55 维（joint_pos 只含 12 个腿关节、joint_vel 含全部 16 个、
动作 = 12 腿位置目标 ×0.25 + 4 轮速度目标 ×10，见元数据 `action_layout`）；
轮足 {go2w,b2w,as2w}_depth_v2 去掉步态相位（53 维，观测项以元数据
`observation_names` 为准），并带低速刹车：指令为零且各轮都慢于元数据
`wheel_brake_rad_s` 时，轮速目标置零（与训练一致）；航向学生（契约名为原契约加
`_heading`，如 go2_depth_v2_heading）观测末尾多一维 heading_error：目标航向在复位后
前两步取机体偏航，之后每个控制步积分整形后的 wz 指令，误差 = 目标 − 偏航（回绕到
±π），限制在 ±0.5 rad（转不动时目标跟着机体，不累积），与训练和 go2-blind 部署一致；
它们按 go2-blind 的指令包络训练（同盲策略）。runtime 加载时按机体核对关节、默认姿态与 PD 增益，不匹配
直接拒绝；默认姿态每个关节差在 0.25 rad 以内时照常加载，机体改用该策略的站姿（复位落地和
稳定站立都按它，`DEFAULT_POSE_TOLERANCE`），所以髋 0 的新盲策略和髋 ±0.1 的感知学生可以在
同一台机体上切换。元数据带 `gait_ref_speed` 的导出（go2-blind 2026-10-27 的腿足定稿）用速度
时钟（`GaitClock`，同 go2-blind 的 mdp.gait_clock 与部署端 gait_phase）：指令下最快那只脚的
速度 |(vx − wz·y, vy + wz·x)|（`gait_foot_xy`）超过 `gait_ref_speed` 时周期缩为
0.6 × ref_speed / 速度；相位复位后为 0，之后每个控制步按当步指令的周期推进；指令低于死区时
读数为 0，原地踏步（`gait_march`，由宿主置 `runtime.march`，默认关）时照常输出。
{go2,as2,b2}_depth_v3 是这批盲策略的深度学生（unitree_rl_mjlab depth-blind，2026-10-27/28），
现为 go2、as2、b2 的感知策略：
观测同航向学生并带速度时钟，深度换成 navcore 真机相机：80×68（竖幅，高×宽）、焦距 38.77 px
（实测 310.15 px 的 1/8，主点居中）、optical Z、不模糊、0.05–3 m 归一化（`DepthSpec`）；近于
0.15 m 的像素读作测不到（`NAVCORE_MIN_RANGE_M`：训练的真实相机随机化每回合在 0.05–0.15 m
里取最近测距，仿真取远端；navcore 的实际值还没测）。元数据
`depth_camera` / `depth_size` / `depth_focal_px` / `depth_kind` / `depth_max_m` 给出画面，
`depth_camera_pos` / `_quat` 给出安装位（base_link，左目 +y 50 mm，水平；CAD 实测：Go2
(0.307, 0.050, 0.127)、AS2 (0.320, 0.050, 0.178)、B2/B2W (0.411, 0.050, 0.158)）。
runtime 加载这类策略时把机体的深度相机移到该位置、按该视场与尺寸另建渲染器，并把相机外壳
挪到 navcore 盒体（镜头之后，看不到）；没有这些元数据的导出照旧用 PIE 相机（60×86、射线距离、
3×3 模糊，装在机型的 `camera_pos`）。现有策略：go2、as2、b2 是 v3 学生，三台轮足是感知
航向学生（unitree_rl_mjlab go2-depth 的 heading 配方，在 go2-blind 的机体上训练，54 维）；
每台另有 go2-blind 的盲策略。被 v3 换下的腿足航向学生（48 维带步态相位）留在 assets/policy/，
不在注册表里：go2_depth_v2_heading，多做了一轮下楼限速微调的 as2_depth_v1_heading_cap、
b2_depth_v1_heading_cap（落脚进度奖励按每步平均速度封顶在指令的 1.4 倍，unitree_rl_mjlab
133244e），以及它们限速前的 model_7000 导出 as2_depth_v1_heading、b2_depth_v1_heading
（下楼约 3× / 4.5× 指令）。旧机体的学生（go2_depth_v2、go2w_depth_v1、as2_depth_v1、b2_depth_v1）
与旧盲策略（go2_blind_v1、as2_blind_v0）也仍在，但不在注册表里：它们的机体已换掉。
被腿足新盲策略换下的 go2_blind_v2、as2_blind_v1、b2_blind_v0 同样留在 assets/policy/。

| 机型 | 感知策略 | 状态 |
|---|---|---|
| go2 | go2_depth_v3.onnx（model_7999） | map3 18 段到达 99.6% / 99.0%（带真机相机误差 98.9% / 98.5%）；平地 0.5 → 0.49 m/s；R2b 下楼 0.4 → 0.34 m/s |
| go2w | go2w_depth_v2_heading.onnx（model_7000） | map3 18 段 99.2% / 99.3%（1.25 倍时限无操作员 99.4%），上楼 1.13×、下楼 0.97× 指令速度；平地 0.5 → 0.56 m/s；R2b 下楼 0.4 → 0.39 m/s；梯上停 4 s 溜车 ≤ 2 cm（上楼途中）/ ≤ 11 cm（下楼途中） |
| as2 | as2_depth_v3.onnx（model_7000） | map3 18 段 99.8% / 99.8%（带真机相机误差 99.5% / 99.9%）；平地 0.5 → 0.49 m/s；R2b 下楼 0.4 → 0.34 m/s |
| as2w | as2w_depth_v2_heading.onnx（model_7000） | map3 18 段 99.7% / 99.8%（1.25 倍时限 99.8% / 99.8%），上楼 1.13×、下楼 0.97× 指令速度；平地 0.5 → 0.52 m/s；R2b 下楼 0.4 → 0.40 m/s；梯上停 4 s：上楼途中最多溜回一级（17 cm），下楼途中 ≤ 1 cm |
| b2 | b2_depth_v3.onnx（model_7000） | map3 18 段 99.4% / 99.7%（带真机相机误差 99.7% / 99.9%）；平地 0.5 → 0.50 m/s；R2b 下楼 0.4 → 0.35 m/s；平地 1.5 m/s 前进 64 台摔 1 台（验收不考这一档） |
| b2w | b2w_depth_v2_heading.onnx（model_7000） | map3 18 段 100% / 100%（1.25 倍时限无操作员 100%），上楼 1.20×、下楼 1.00× 指令速度；平地 0.5 → 0.59 m/s（偏快约 20%）；R2b 下楼 0.4 → 0.42 m/s；梯上停 4 s 溜车 ≤ 12 cm（上）/ ≤ 4 cm（下） |

map3 18 段 = unitree_rl_mjlab `scripts/go2_depth/eval_map3.py --set flights`（go2-blind 的
18 段上下楼，每段 64 台，限时内爬升或下降到梯段高度的 85%、且此前没摔算到达），按 go2-blind
的验收设置测：时限是按指令速度走完所需时间的 2 倍（`--time-slack 2.0`，即 go2-blind
`scripts/go2_blind/eval_map3.py --time-scale 1.6`；unitree_rl_mjlab 617d769）；斜线前后为
无 / 有操作员（扶航向、回中线），无操作员时上下楼指令都是 0.5 m/s（`--no-operator --down-vx 0.5`），
有操作员时上楼 0.5、下楼 0.4 m/s。v3 三行是 depth-blind 的验收检查，直接跑 go2-blind 的 map3
脚本（`--time-scale 1.6`，同一设置），括号里是加上训练里的真机相机误差（`-Real` 任务：延迟、
噪声、缺测、最近测距等）的到达率。轮足三行括号里是 1.25 倍时限、上楼 0.5 / 下楼 0.4 m/s 的
到达率。到达之后的摔倒不计（2 倍时限走得更远，as2w 无操作员有 3.4% 走过梯段后撞上尽头的墙）。
速度倍数取上楼 0.5、下楼 0.4 m/s 时。
其余数字为本仓 smoke.py（flat、down = R2b 8 级）与梯上停车测试，均 0 摔。

腿足盲策略（go2-blind 2026-10-27 的分阶段定稿，所有验收项通过；由
`scripts/import_blind_policy.py` 从其 policy.onnx + deploy.yaml 导入，带速度时钟与 gait_march）：
go2_blind_v3（Go2 phased_polish model_8000）、as2_blind_v2（As2 phased_polish model_10000）、
b2_blind_v1（B2 phased_polish model_7000），髋默认角均为 0。本仓 smoke（无摔）：平地 0.5 →
0.50 / 0.48 / 0.50 m/s，站立漂移 ≤ 0.007 m/s，R2b 下楼 0.4 → 0.38 / 0.37 / 0.38 m/s。

轮足盲策略（go2-blind 4fc13b4 deploy/robots/<robot>/config/policy/velocity/blind_v0）：
go2w_blind_v0（9497）、as2w_blind_v0（8200）、b2w_blind_v0（8700）；smoke flat / down / map3
均 0 摔。此前的腿足盲策略 go2_blind_v2（Go2 model_23400）、as2_blind_v1（9500）、b2_blind_v0
（8999）同批。go2-blind 自测 map3 18 段（同上面的验收设置，无 / 有操作员）：Go2 87.4 / 95.1%、
As2 85.3 / 99.9%、B2 93.3 / 95.1%、As2W 81.0 / 100.0%。

网页操作面板（`webpanel.py`，view/hil 共用，在页面右上角控制面板最上方）：

- **Map**（下拉）：view.py 运行中切换地图（`plant.SCENES` 注册表顺序：map3 楼梯井、
  map2 室内、flat 平地）：当前机型和策略在新地图的默认出生点重建，所有网页的相机
  重新对准机器人；失败则保持原地图并弹提示。新地图里第一次切机型要现编（约 1 s，
  启动时只预编当前地图的机型）。`--preview` 时不提供（splat 管线装的是启动地图），
  hil.py 只显示（`--scene` 启动时选）。出生高度按模型所属的地图算
  （`Bindings.scene_name`）。
- **Robot**：view.py 运行中切换机型（重建机体、策略、web 场景，失败则保持原
  机型并弹提示）；hil.py 只显示（`--robot` 启动时选，契约面在启动时测相机
  安装位）。
- **Perception**：感知策略 ↔ 盲策略原地热切（机体状态保留，策略记忆/历史清空）。
- **Gamepad**：手柄接管开关（与手柄 A 键双向同步）。
- **Wall guard**（默认开）：防撞墙（`sim/guard.py`）。策略深度图里正前方有墙时，
  前进速度上限 = 1.5 ×（到墙距离 − 0.45 m），机器人减速并停在墙前约 0.5 m（相机
  在机身最前端）；后退、转向、横移不限，状态行写「前方墙 x.xx m，限速」。墙按高度
  和楼梯区分：机身宽度内、相机上下 −0.10~+0.20 m 的高度带里，同一距离（5 cm 内）
  的点要跨 0.22 m 高才算墙——台阶立面最高 0.19 m，下一级隔着一个踏面，所以楼梯
  不触发（map3 R2b 上下楼实测 0 次误限速）。原因：深度学生把正前方的墙当台阶去爬
  （被推着顶墙时 Go2 仰到 29°、AS2 / B2 会摔），停在 0.3 m 仍会仰头，0.45 m 起三台
  都稳（横滚 ≤ 2°、俯仰 ≤ 6°）。只对 navcore 学生（v3）生效：PIE 相机朝下 20°，
  看不到相机上方。hil 里只管键盘和手柄，栈的 /cmd_vel 不限。
- **Track robot**（默认开）：视角跟着机器人：场景按机身位置平移、按朝向反向旋转
  （0.15 s 低通；复位、摔倒这类 ≥ 1 m 或 ≥ 1 rad 的跳变直接跟上），机器人始终在视图
  原点，转身时视角相对机身的角度不变。网页自带的鼠标操作照常：左键拖动绕它转、滚轮
  缩放、右键平移。新打开的网页（以及切换地图后）从机器人正后上方看（1.8 倍机身长，
  俯角 30°）。相机若会隔着地板或梯段看机器人，就拉到遮挡物前面（留 0.15 m，最近
  0.3 m），没有遮挡后 0.3 s 内缓回。关掉后场景不再动、相机停在原处，用鼠标或
  W / S 前后、A / D 左右、Q 下降移动视角（先点一下网页）。mjviser 自带的
  Track camera 勾选框已去掉：它把场景平移到原点再对准，和这里的跟随打架。
- **Keyboard speed**：键盘按住时的速度占当前策略训练指令范围的百分比
  （10–100%，默认 50%），切换机型或策略后按新范围算；面板里写着各方向的实际
  速度和视角按键（键位见下文「键盘」）。
- **Max accel / Max yaw accel**：键盘和手柄指令的最大加速度（`sim/accel.py`）：
  vx、vy 合起来每秒最多变 1.0 m/s（默认，可调 0.2–3.0 m/s²），wz 每秒最多变
  2.0 rad/s（0.5–6.0 rad/s²）；加速、减速都限，手柄 B 急停除外（立即归零）。
  按住 ↑（50% 时 0.75 m/s）约 0.75 s 加到全速，松开同样斜坡减到零。
- **Reset to spawn / Stop**、**Policy depth**：策略当前深度输入（PIE 60×86，
  navcore 学生 80×68 竖幅；10 Hz，turbo：近红、3 m 外深蓝；navcore 近于 0.15 m
  的像素按测不到处理、也显示深蓝，同训练里的真实相机随机化），仅感知策略时显示；底部状态行（策略 · 指令 ·
  手柄 · 踏步）。感知策略驱动时，它的深度相机画在机身上（小金字塔：顶点在镜头，
  张角 = 策略看到的范围）。
- **Hip temperature (est.)**：四个胯关节电机绕组温度的估算（`sim/thermal.py`：
  仿真关节力矩 → 铜损发热 → 经外壳散到 25 °C 空气；参数按电机大小估，未在真机上
  标定），最近 20 秒的实时曲线，换机型重来。

所有宿主（view / hil / dog / mj / smoke）都用注册表策略（go2 =
go2_depth_v3）；`--policy <onnx>` 固定单个 ONNX，`--blind`（view）以盲策略
启动。没有 `contract` 元数据的 ONNX（如旧 PIE 导出）会被拒绝。view.py 的
splat 管线只在 `--preview` 时建立（它是唯一消费者），所以纯网页形态不再需要
gausscam / splat 资产。

## HIL：DDS 契约面

同一套宿主循环（hil.py）+ 同一份契约（hil/__init__.py），单一路径：rclpy
节点（/insight_full）把契约直发 DDS——与 tinynav 栈同域（host network），
节点身份即真（pilot 的就绪检查按名字找到的就是发布器本身）。跑在 mjsim
容器里（jazzy，py3.12+rclpy；宿主 sim venv 无 ROS python，launch.sh 会
拦截并指路容器）。跨机形态（网线连 Orin Nano 跑 pilot 全栈）的环境配置、
编址与实测开销另见 [hil-orin-pilot.md](hil-orin-pilot.md)。

契约（对齐 `../docs/looper-contract.md` oracle + 扩展）：
infra1/infra2 整流灰度 mono8 544×640、depth mono16mm、color JPEG（四条 + vio_image
ExactTime 同 stamp，10Hz 默认）、camera_info×3、vio_100hz 100Hz、vio_status 1Hz
latched、tf_static 5s 重发、imu ~200Hz BEST_EFFORT。**stamp = 采集时刻位姿的
采样时刻**；IMU = imu site 物理传感器每个物理子步（200Hz）采样，vio/imu 完整
包络相机 stamp（补采补丁已删）。

**搭链路（对端同样要钉）**：mjsim 侧把 Cyclone 钉在 loopback 单播（镜像已
烤 `tools/probes/cyclone_localhost_unicast.xml`）；对端不是 Cyclone+同一钉定
就连不上或掉帧——Fast DDS 订阅端会把 544×640 图像从 10Hz 掉到 ~4Hz，不钉 URI
则 0Hz（连发现都没有）。对端固定动作 = `source tools/probes/hil_peer_env.sh`
（非覆盖式：已设且有效的 URI 不动，分机 launch 自设的配置照常生效）；**rig
容器已默认接线**（`docker/shell-env.sh` + 容器 `/root/.bashrc`，后者容器
recreate 后需重加）。实测 `tools/probes/hil_rate_check.py`（rclpy 真订阅，
别用 `ros2 topic list`——本环境会假阴）。

## 启动

```bash
cd tinynav-sim
bash docker/run-mujoco-sim.sh          # 默认 hil：DDS 直发 + viser :8012 + 键盘遥操
bash docker/run-mujoco-sim.sh dog -- --profile go2 --iface <nic> --contract
                                       # 全栈 emu（dog.py）：viser :8012 同样默认起（--view none 关）
bash mujoco/scripts/launch.sh sim      # 交互沙盒（会话 sim；viser :8012，--preview 加 MJPEG :8888）
bash docker/run-mujoco-sim.sh sim --robot go2w --gamepad
                                       # 同上，容器内；--robot 起始机型、--gamepad 手柄默认接管
bash mujoco/scripts/launch.sh mj       # 物理排查（会话 mj，原生窗口）
# hil 需要 rclpy —— 宿主 sim venv 无 ROS python，launch.sh 会拦截并打印
# 容器路径；宿主 hil 会话同理。--view glfw 换原生窗、--no-viewer 关窗。
tmux kill-session -t sim|mj            # 结束宿主会话
```

键盘（hil/sim 都是 /dev/input 全局捕获，不依赖窗口焦点）：
按住 ↑ 前进、↓ 后退、←/→ 左右转、`,`/`.` 左右横移（vy）；按一下 M 开 / 关原地
踏步（零指令时踏步而不是站着；只对训练过的策略生效——腿足新盲策略，元数据
`gait_march`；Stop / Space、复位会退出踏步）。速度 = 当前策略训练
指令范围在该方向的端点 × 面板的 **Keyboard speed**（默认 50%）：盲策略（vx
−1.0~1.5、vy ±0.6、wz ±1.5）在 50% 时前进 0.75、后退 0.5、横移 0.3、转向 0.75，
深度学生（vx −0.5~1.0、vy ±0.5、wz ±1.2）是 0.5 / 0.25 / 0.25 / 0.6；没有策略
范围的宿主（dog）仍用固定的 0.5 / 0.2 / 0.3 / 0.3。松开后先
补发 1s 零速（flush，对齐 gazebo teleop），然后命令源转 idle。sim/mj 里 idle
只反映在状态行（控制环仍喂零）；hil 里键盘是**覆盖层**——live/flush 键盘赢、
idle 后 DDS /cmd_vel 恢复驱动（人手微调不抢栈的导航权）。hil 无 Space/R 键
绑定：Reset/Stop 只在 web viewer 按钮上，误按 R 永远不可能复位一个在跑的
HIL 会话。

手柄（sim/gamepad.py，同样读宿主 /dev/input，USB/蓝牙均可，容器经
run-mujoco-sim.sh 的 /dev/input 挂载可见；插拔 2s 内自动识别）：左摇杆
vx（推满 = 训练包络上限 1.0，后拉 0.5）/vy（±0.5），右摇杆 X wz（±1.2），A
开关手柄接管，B 按住急停（零指令，同时退出踏步），X（或 Y：不同驱动 / 模式下
X 键报 BTN_X 或 BTN_Y）开 / 关原地踏步。接管开（`--gamepad`、面板勾选或按 A）
时手柄驱动——摇杆回中即原地站立，不交还；hil 里手柄覆盖 DDS /cmd_vel，键盘
再覆盖手柄。包络随当前策略的训练指令范围。

键盘和手柄的指令都过最大加速度限幅（`sim/accel.py`，面板 **Max accel** /
**Max yaw accel**）：按键或推杆给出的速度按斜坡逼近，不再一步跳变。hil 里只限
键盘和手柄：栈的 /cmd_vel 不限——键盘 / 手柄交还后，指令先按同样的加速度
追上栈的指令，之后原样透传。

## 依赖

venv 需要：mujoco≥3.2、numpy、onnxruntime、opencv（headless 即可）、wgpu≥0.32、
gausscam（原 splatsense；mjsim 镜像按 wheel 烤入。宿主 venv 直跑走
launch.sh——它自动发现同级 splatsense checkout 挂 PYTHONPATH，无需 pip；
或直接 `pip install gausscam`）。
hil 另需 rclpy（mjsim 镜像内）；`--view web` 另需 viser（镜像已带）。

HIL 验证与故障记录：`../docs/looper-contract.md`（oracle 契约）、
`mujoco/hil/selftest_ros.py`（ros 面自检）、
gausscam 集成期档案（splatsense 开源前存档，含 p3-pie-locomotion.md §6.5
渲染栈早期验证与故障；不入库，本机存于 `~/workspace/dm/archive/gausscam-archive/`）。

## 跨厂 GPU 兼容性实测（选 Vulkan 的初衷验证）

hil 首参即 GPU 选择器（`hil.py intel ...`，子串匹配适配器名）。容器需
Intel Vulkan 驱动（ANV）：基座 Ubuntu24.04 的 Mesa 24.0 **不认识 ARL 核显**
（装了也枚举不到），需 kisak-mesa PPA 升到 Mesa 25.2 后 ANV 正常枚举——
镜像若要支持新 Intel iGPU，应升级 Mesa 或在文档注明。

实测（544×640 双目 10Hz 契约，1.6M 高斯，optcol 变体自动切换）：

| GPU | render/帧 | cam 实际 | 结论 |
|---|---|---|---|
| RTX 5070 (opt) | 28-43ms | 10Hz ✓ | 基准 |
| RTX 2060M (opt) | 66-77ms | 10Hz ✓ | Turing 够用 |
| Intel ARL iGPU (optcol) | 131-200ms | 3-7Hz ✗ | 能跑，扛不住 10Hz 契约 |

跨厂端到端（wgpu splat + optcol 排序 + DDS 契约面）在 Intel 上完整工作，
pick_variant 的 Mesa 分型按设计生效。Intel-only 部署需降 `--cam-hz` 或
降分辨率；HIL 契约速率仍需 NVIDIA/Turing 级。

## 渲染解耦与合成基准

**hil 双线程**：控制线程 200Hz 物理+策略+IMU（substep 采样），每 tick 把
13 link/相机位姿写入快照槽（带锁，微秒级）；渲染线程按 `--cam-hz` 节拍
取快照 → wgpu 渲染 → JPEG → 发布（publish 线程安全，同 ros-fast-pub 模式）。
渲染慢只掉帧率、**永不阻塞物理/IMU**（Intel 实测：cam 4.7-5.9Hz 时 imu
132/s，重构前 30-40；NVIDIA 上 imu 200-204 满速）。旧"相机 stamp 补采 IMU"
补丁已删（不再有窟窿）。preview 的场景视图在渲染线程用 mjData 副本渲染。

**gausscam 合成基准**（跨 GPU 一条命令）：
```bash
python -m gausscam.bench --device-sub intel --n 200000 --frames 30
```
固定种子 host 侧合成（输入跨设备逐位一致）+"轴向地标"解析深度锚点
（±5% 校验）。实测：5070 4.56ms p50 / ARL iGPU 45.6ms（10×），两家
rgb_sum 逐位一致。注意三件事：①kernel 吃**激活值**（scale 米制、
opacity 0..1——对齐真实 npz；INRIA PLY 原始值需自行 exp/sigmoid）；
②合成布局模拟真实统计（远墙小 splat），大重叠 blob 会超 (gaussian,tile)
对数预算（~4M）直接死锁，layout="stress" 仅作驱动健壮性探针；
③Intel ARL ANV 对小 dispatch 有崩溃 bug（n≤20k abort、200k 正常），
且 opt 变体 OneSweep 在 Mesa/Intel 死锁（pick_variant 自动避）——
跨厂商场景一律用返回的变体。

## 现状与待办

- 双机（本机 5070 / 远端 2060M）mjsim 镜像均验证通过，hil 双线程版为当前
  形态；渲染链已在 gausscam 0.1.1+ 的 static-scene 模式下运行（场景直接
  `from_ply`，狗体不再参与渲染）。
- 待办：012 navcore 侧 HIL 验证阶梯未跑；Intel 若转正需镜像层升 Mesa
  （kisak PPA 25.2）+ `--cam-hz` 降档。
