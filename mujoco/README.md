# MuJoCo dog sim: 策略驱动 + wgpu 3DGS 传感器视图 + HIL 双传输面

Unitree Go2 由楼梯下坡策略（ONNX）驱动，在 map3 楼梯井里行走（MuJoCo 物理），
狗眼相机画面由 gausscam（原 splatsense）的 wgpu 管线渲染（3DGS）。整体跑在
tinynav-sim 仓，作为"整体仿真"侧：既可以人机交互遛狗，也可以作为传感器源喂
navcore 做验证。

## 文件架构

```
mujoco/
├── README.md            本文件
├── view.py              交互宿主：viser 网页 viewer（默认）+ splat；
│                        --preview 才开 MJPEG 推流
├── physics_view.py      纯物理 viewer 宿主（无 splat）——姿态/物理排查
├── smoke.py             headless 验证：flat | map3 | down
├── hil.py               HIL 宿主入口：rclpy 契约面直发 DDS（无桥、单传输）
├── sim/                 仿真核心包（与传输/显示无关）
│   ├── contract.py      策略 I/O 契约（vendor 自上游 sim2sim，常量勿改）
│   ├── policy.py        ONNX 策略会话（GRU memory + 元数据契约校验，CPU 单线程）
│   ├── expert.py        双专家选择器（stairs+flat；vx 迟滞自动切 + CLI 强制档，
│                        GRU 切换清零）
│   ├── plant.py         世界搭建：map3 场景 + 机器人 + 执行器内 PD + rig 相机
│   ├── runtime.py       50Hz 控制循环（深度 10Hz → obs → 策略 → 4×mj_step）
│   └── keyboard.py      /dev/input 全局按键轮询（零依赖）
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
│   ├── go2/             go2_pie.xml（PIE 策略变体骨架；网格在 model/ 资产包，见下）
│   └── policy/          policy.onnx（= model_18997，PIE 楼梯策略，见
│                        ../docs/stairs-map3-experiments.md）
│                        + policy_24000_flat_omni.onnx（平地全向专家，
│                        view.py 双专家 auto 的后退分支，见同文）
│                        + policy.sha256（训练侧校验清单）
└── scripts/
    ├── launch.sh        统一 tmux 启动器（sim | mj | hil）
    └── probe_viewer.py  X/GLFW 健康探测
```

资产来源：**不入 git 的大资产统一走 `model/` 资产包**——把 `model.zip`
解压到仓根即得（`model/splat/` = 场景，`model/go2/` = 减面网格；包内自带
`VERSION.txt` 与 `MANIFEST.sha256`，解压后 `sha256sum -c MANIFEST.sha256`
验包）。运行期查找链：环境变量 `ROBOT_ASSETS` → 仓内 `model/` → 旧
`~/workspace/dm/robot-assets` checkout；mjsim 镜像把整个包烤在镜像内
仓根（与宿主机解压后布局一致），构建前放好 `model/` 即可。包内容出处：
`map3_scene.ply`
导出自 gsplat 侧 map3 建图 3DGS 场景（`examples/map3_stairs/assets_local/
map3_scene`，INRIA 布局，`from_ply` 直读 RAW 值）；`map2_scene.ply` 为
MetaCam map_2 的 spirula 训练输出直拷（step-50000，2.98M 高斯，706MB）；
`go2/` 为 robot-assets 仓的
减面二进制 MSH（1.3MB，出处链/重生成方法见该仓 README——该仓本地
checkout 尚无提交，网格版本以 MANIFEST 校验和为准）。
历史：本仓 `assets/go2/assets/` 曾带 25.5MB menagerie OBJ，
减面迁移后删除（上游 menagerie 与 splatsense examples 各有全量副本）。
入库的 `policy.onnx` 来自 RL 训练仓（hash 见 policy.sha256，
其 .pt 血缘在训练侧，本仓只带 onnx）。

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
| sim | `view.py` | viser 网页 viewer + 狗眼 splat（`--preview` 才推流），无 DDS | 人看/遥操 |
| mj | `physics_view.py` | 纯 viewer + 策略 | 物理排查 |

可视化四通道：viser 网页 viewer（hil/sim 都默认；无 DISPLAY 依赖、CPU 约为
GLFW viewer 的 1/17，带浏览器 Reset/Stop 按钮——hil 里 Reset/Stop 只有按钮，
无键盘绑定）；MuJoCo 原生窗口（`--view glfw` 回退项，需 X）；浏览器 MJPEG
推流（`--preview` 显式开启，sim/hil 都是 :8888，第二屏看狗眼用）；rqt/ROS
工具直接订阅契约话题。
OpenCV/Qt 窗口形态已废除（混合图形机器上 Qt 是四客户端
段错误的成员之一；详细档案不入库，本机存于
`~/workspace/dm/archive/gausscam-archive/p2-wgpu-report.md` §7/§8）。

场景：`plant.SCENES` 注册表，hil/sim/mj 都吃 `--scene map3|map2`（默认
map3）。**map3** = 楼梯井（123 box 精确碰撞，出生 F1 正对 L0 梯段，hil 载
楼梯专家 18997）。**map2** = MetaCam map_2 室内（spirula 50k，2.98M 高斯；
平地 box + 边界墙碰撞，出生 `low` = gsplat 验证过的低采段点 (3.21,−5.54)，
yaw −11.5°——狗眼 0.45m 只有低采段渲染干净，**换出生点先 `--preview` 渲两
帧对质再走**；hil 载平地专家 24000，sim 双专家 auto 不变）。碰撞地面高
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

## HIL：DDS 契约面

同一套宿主循环（hil.py）+ 同一份契约（hil/__init__.py），单一路径：rclpy
节点（/insight_full）把契约直发 DDS——与 tinynav 栈同域（host network），
节点身份即真（pilot 的就绪检查按名字找到的就是发布器本身）。跑在 mjsim
容器里（jazzy，py3.12+rclpy；宿主 sim venv 无 ROS python，launch.sh 会
拦截并指路容器）。

契约（对齐 `../docs/looper-contract.md` oracle + 扩展）：
infra1/infra2 整流灰度 mono8 544×640、depth mono16mm、color JPEG（四条 + vio_image
ExactTime 同 stamp，10Hz 默认）、camera_info×3、vio_100hz 100Hz、vio_status 1Hz
latched、tf_static 5s 重发、imu ~200Hz BEST_EFFORT。**stamp = 采集时刻位姿的
采样时刻**；IMU = imu site 物理传感器每个物理子步（200Hz）采样，vio/imu 完整
包络相机 stamp（补采补丁已删）。

## 启动

```bash
cd tinynav-sim
bash docker/run-mujoco-sim.sh          # 默认 hil：DDS 直发 + viser :8012 + 键盘遥操
bash mujoco/scripts/launch.sh sim      # 交互沙盒（会话 sim；viser :8012，--preview 加 MJPEG :8888）
bash mujoco/scripts/launch.sh mj       # 物理排查（会话 mj，原生窗口）
# hil 需要 rclpy —— 宿主 sim venv 无 ROS python，launch.sh 会拦截并打印
# 容器路径；宿主 hil 会话同理。--view glfw 换原生窗、--no-viewer 关窗。
tmux kill-session -t sim|mj            # 结束宿主会话
```

键盘（hil/sim 都是 /dev/input 全局捕获，不依赖窗口焦点）：
按住 ↑ vx +0.5、↓ vx −0.2、←/→ wz ±0.3；松开后先补发 1s 零速（flush，对齐
gazebo teleop），然后命令源转 idle。sim/mj 里 idle 只反映在状态行（控制环
仍喂零）；hil 里键盘是**覆盖层**——live/flush 键盘赢、idle 后 DDS /cmd_vel
恢复驱动（人手微调不抢栈的导航权）。hil 无 Space/R 键绑定：Reset/Stop 只在
web viewer 按钮上，误按 R 永远不可能复位一个在跑的 HIL 会话。
双专家选择器（sim/mj）默认 auto：↓ 自动切平地专家、其余指令走楼梯主力，无需任何
按键操作（`--expert stairs|flat` 可强制单边）；切换时目标专家 GRU 清零，
状态行 `exp=` 显示当前激活专家。

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
