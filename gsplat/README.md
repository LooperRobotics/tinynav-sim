# gsplat/ — 3DGS（gs_playground）仿真层

照片级仿真：MotrixSim 物理 + 3D Gaussian Splatting 渲染 + RL 步行策略，
传感器面与 `gazebo/`（gzsim）**逐字节一致**（544×480、fx=fy=272、51 mm
基线、200 Hz IMU、同 frame_id），tinynav 栈无法区分两个仿真。从宿主+容器
分体形态收进本仓、收进单容器。进度与验收记录见
`docs/gsplat-sim-progress.md`。

## 架构（同一容器，两个进程）

```
[gs-sim] sensor_server（/opt/venv_gs python）
    MotrixSim 物理 500 Hz + go2 RL 策略(onnx, CPU) + gsplat 3 相机渲染
      │ 写 /dev/shm/gsplay_sensors.bin（无锁环形缓冲 v2）
      ▼
[bridge] gs_ros_bridge（系统 python3 + rclpy）
    → /camera/camera/{infra1,infra2,color}/... + camera_info
    → /camera/camera/imu、/sim/gt_pose、/clock
    ← /cmd_vel → /dev/shm/gsplay_cmd.txt → RL 策略
      ▼ 话题面与 gazebo 完全一致
[stack] reference py 栈（--stack full）或 tinynav_cpp（--stack cpp）
```

保留环形缓冲而不是进程直发 DDS：渲染循环（15 Hz 预算 66.7 ms、渲染
~35 ms）不能被订阅方节奏拖住，且 ring 的重启检测/防撕裂协议已踩完坑。

## 一次性环境 bootstrap（容器内）

```bash
# 宿主机执行（脚本会 docker exec 进容器）：
bash gsplat/setup_env.sh
```

创建 `/opt/venv_gs`（torch 2.7.0+cu128 / gsplat 1.5.3 / motrixsim-core 私有
索引 / onnxruntime，与 ROS 栈的 /opt/venv 完全隔离——numpy 2.x 不进 ROS
venv）、`/opt/cuda-12.8` nvcc（容器自带 CUDA 12.2 不支持 RTX 5070 的
sm_120，gsplat JIT 需要 ≥12.8）和 `/opt/cuda-shim-gs`。回滚 = 删这三个目录
+ `/root/.cache/torch_extensions`。运行期必须保留 `CUDA_HOME` 和 `PATH`
（gsplat import 时检查 nvcc，否则不加载已编译 kernel）——run_gsplat.sh 已内置。

## 日常使用（容器内）

```bash
docker exec -it tinynav bash
cd /workspace/dm/tinynav-sim

# 传感器面 + C++ 栈（建图模式）：teleop 开狗，/mapping/start + /mapping/stop 出图
bash gsplat/run_gsplat.sh --stack cpp

# 带图导航（map format v2）
bash gsplat/run_gsplat.sh --stack cpp --map --map-dir output/map_gs_church

# 发导航目标（建图系；教堂图 = 出生直行方向，与 gazebo/scene/pub_target.sh 同格式）
ros2 topic pub -w 1 -r 5 -t 5 /control/target_pose nav_msgs/msg/Odometry \
  "{header: {frame_id: world}, pose: {pose: {position: {x: 10.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}}"

# 剧本直行（12 m，真值航向保持）——录包/建图免手操
bash gsplat/run_gsplat.sh --stack cpp -- --drive-start 5 --drive-distance 12 --drive-speed 0.5

# 只起传感器面（pilot/其他消费者拥有栈）；或 python 栈
bash gsplat/run_gsplat.sh --stack sensor
bash gsplat/run_gsplat.sh --stack full

# 开工前纪律：一键确认狗状态（真值+yaw+SLAM odom+IN/OUT）
bash gsplat/gs_state.sh --slam --map-dir output/map_gs_church

# MotrixSim 碰撞+狗窗口（独立进程；默认不开，需要时显式加）
bash gsplat/run_gsplat.sh --stack cpp --gui
# rviz 默认就起（与 gazebo/run_simulator.sh 一致）；想要满 rtf 可关掉
bash gsplat/run_gsplat.sh --stack cpp --no-rviz

# 收摊
bash gsplat/kill_gsplat.sh
```

场景：`--scene church`（教堂中殿，出生朝 +Y 沿过道）或 `--scene nav1`。

**窗口默认情况**：rviz 默认启动；**MotrixSim viewer（`--gui`）默认关闭**——它渲染
走 Mesa llvmpipe，实测开它会挤掉相机流的预算（52→75 ms/帧）。两者都可以显式控制
（`--gui` / `--no-rviz`，launch 的 `gui:=true` / `rviz:=false`）；细节见「可视化」节。

## 资产依赖与路径配置（不进 git 的部分）

**没有任何 ply / MJCF / onnx 进 git**（教堂场景 119 MB、go2 网格 27 MB……）。
所有资产从**外部 gs_playground checkout** 读取，靠一个环境变量定位：

```
GS_PLAYGROUND_ROOT = <gs_playground 检出目录>
  容器默认: /workspace/github/simulation/gs_playground
  宿主:     /home/dm/workspace/github/simulation/gs_playground
```

`gsplat/configs/*.json` 里的路径**全部相对 `<GS_PLAYGROUND_ROOT>/demo/navigation`**
解析（`sensor_server.py` 的 `resolve_gs_path()`；rig 例外，见下）：

| config 字段 | 解析结果 | 大小 |
|---|---|---|
| `scene`（如 `church_scene/mjcf/scene.xml`） | 场景目录 `church_scene/`（MJCF + 其 `<include>` 兄弟文件 + meshes + 3dgs 一起要） | 119 M |
| `scene_gaussians.scene`（如 `church_scene/3dgs/church.ply`） | 3DGS 高斯 ply | 119 M（含在上面） |
| `robot_gs_dir`（`assets/go2`） | 机器人各 link 的高斯 ply（13 个） | 9.4 M |
| （隐含）`models/robots/navigation/go2/` | go2 MJCF + `assets/*.obj` 网格 | 27 M |
| （隐含）`policies/go2_policy.onnx` | RL 步行策略 | 755 K |
| （隐含）`configs/asound-null.conf` | ALSA null 配置（消音频初始化噪声） | 31 B |
| rig `go2_sensor_rig.xml` | **本仓生成**，不是拷贝：`python3 gsplat/robots/go2/make_sensor_rig.py`（必须生成在 go2 MJCF 同目录——网格按 MJCF 所在目录解析） | 15 K |

**校验（推荐先跑这个）**：

```bash
python3 gsplat/tools/check_assets.py           # 逐项 OK/MISSING + 体积，末尾给打包命令
```

它会按上面的规则解析每个 config 的依赖、报告缺失项，并打印**打包给同事的命令**：

```bash
# 打包方（church + nav1 + go2 全套 = 174.6M 未压缩；只带 church 时 156.5M）
tar -C <checkout>/demo/navigation -czf gsplat_assets.tgz \
    assets/go2 church_scene configs/asound-null.conf \
    models/robots/navigation/go2/assets \
    models/robots/navigation/go2/go2_mjx.xml \
    models/robots/navigation/go2/go2_sensor_rig.xml \
    nav_scene_1 policies/go2_policy.onnx
# 接收方：解到自己的 gs_playground checkout，再复核（rig 已在包里；改过
# go2_mjx.xml 才需要重跑生成器）
tar -C <checkout>/demo/navigation -xzf gsplat_assets.tgz
python3 gsplat/robots/go2/make_sensor_rig.py    # 可选：重新生成 rig
python3 gsplat/tools/check_assets.py
```

### 现成的资产包

```
output/gsplat_assets.tgz        141 MiB（解包 174.6 MiB）
  md5  fd0384dbca8450621f3768c12086b677
  内容: church_scene/ nav_scene_1/ assets/go2/ models/robots/navigation/go2/
        {assets,go2_mjx.xml,go2_sensor_rig.xml} policies/go2_policy.onnx
        configs/asound-null.conf
```

`output/` 是 gitignored，包不随仓库走——**拷给同事时连 md5 一起给**，接收方：

```bash
md5sum gsplat_assets.tgz                       # 应等于上面的值
git clone <tinynav-sim> && cd tinynav-sim      # 代码走 git，包里只有资产
tar -C <TA 的 gs_playground>/demo/navigation -xzf gsplat_assets.tgz
export GS_PLAYGROUND_ROOT=<TA 的 gs_playground>
python3 gsplat/tools/check_assets.py           # 应 ALL PRESENT
bash gsplat/setup_env.sh                       # 装 venv_gs + CUDA（宿主机执行）
bash gsplat/run_gsplat.sh --stack sensor       # 冒烟：话题面 + rviz
```

包已在**独立解包目录**做过两层验证：`check_assets.py --scene all` 全绿 + 用
`GS_PLAYGROUND_ROOT` 指向解包目录真跑 `sensor_server --out`（教堂 766 万高斯、
15.2 Hz 相机、200.2 Hz IMU、渲出 3 张图）。

`check_assets.py` 默认输出不带 rig xml（它把 rig 标成"生成物"）；手工交接时把 rig
一起打进去可以省接收方一步（生成器需要 venv_gs 的 numpy/scipy 才能跑）。

换机器/换路径只改 `GS_PLAYGROUND_ROOT`（run_gsplat.sh 与 launch 都读它；也可直接
`GS_PLAYGROUND_ROOT=/path bash gsplat/run_gsplat.sh ...`）。新增场景 = 上游 checkout
里加一份场景目录 + 在 `gsplat/configs/` 加一个 json。

## 分机部署（x86 仿真面 ↔ Orin 栈）

与 `gazebo/launch/sim.launch.py` 同构的独立 launch：`gsplat/launch/
gsplat_sim.launch.py`。桥发的话题面与 gz 完全一致，**Orin 侧
`orin_stack.launch.py` 零改动**（链路对岸无法区分两个仿真器）。launch 自带
完整 env（CycloneDDS 钉 USB 网卡 + venv_gs/CUDA shim + remote_planning
重映射），不需要 tmux 手动重导环境：

```bash
# x86 站点（rig 容器）：仿真面（server+桥+跟随器）+ 感知
ros2 launch gsplat/launch/gsplat_sim.launch.py remote_planning:=true
ros2 launch tinynav_cpp perception.launch.py

# Orin 站点（Humble，宿主机；分机环境一键 source）
ssh nvidia@192.168.55.1
source /home/nvidia/nav_env.sh        # Humble + RMW=cyclone + XML 钉 l4tbr0
ros2 launch tinynav_cpp orin_stack.launch.py map_path:=<v2 图目录>
```

链路验收记录（USB NCM 直连，话题级）：Orin 可见全部 13 个仿真面话题，
infra1 图像 15 Hz 稳定过链、IMU 200 Hz、GT 50 Hz；x86 perception 起后
`/camera/camera/slam/*` 8 个出站话题在 Orin 全部可见（odometry_visual ~5 Hz）；
反向 `/cmd_vel` 从 Orin 发布 → 狗行走验证通过。

## 可视化

设计取舍：`sensor_server` 是**无窗口**服务（渲染走 CUDA 光栅化，不建 GL 窗口——
上游 demo 的 RenderApp 是独立窗口应用，无法 attach 到运行中的会话，且其主视口
显示的是物理网格、3DGS 只在角落小面板）。看会话用 ROS 侧 viewer，容器
`DISPLAY=:1`（X11 已挂载）：

```bash
# 狗的第一视角（3DGS 渲染原图——最直接的"画面"）：
ros2 run rqt_image_view rqt_image_view          # 下拉选 /camera/camera/color/image_raw
                                                #（infra1 看灰度/VIO 输入）
# 导航视图（占据/轨迹/odom + GT）：run_gsplat.sh 自带的 rviz 窗口，
# 或手动： rviz2 -d docs/vis.rviz   （/sim/gt_pose 可作为 Odometry 显示加进去）
```

第三人称窗口（上游 demo 那种）：`--gui`（run_gsplat.sh / launch 的 `gui:=true`）。
主视口=碰撞网格 + 狗（鼠标轨道/缩放），**独立进程** `gsplat/tools/view_window.py`，
读 server 写的状态槽（`/dev/shm/gsplay_state.bin`：sim_time + 完整 dof_pos @50Hz，
浮动基位姿含在内）驱动自己的 MJCF 副本——窗口显示的就是活会话本身。

**为什么独立进程**：容器里 wgpu 只有 Mesa llvmpipe（无 NVIDIA GLX/Vulkan ICD；
NVIDIA EGL 下 wgpu 直接 panic，所以 viewer 与 server 都会主动摘掉
`/root/.bashrc` 烤入的 PRIME/EGL 三件套），视口每帧要花真实 CPU；放进仿真进程会
抢主线程。实测对比（church，含桥/控制/rviz 全家桶）：

| 模式 | 仿真 rtf | 窗口帧率 |
|---|---|---|
| `--gui`（独立 viewer，默认） | **0.91–0.92** | **27–30 Hz** |
| `--gui-embed`（进程内嵌，遗留） | 0.68 | 15 Hz |
| 视口尺寸缩小（640×360 等） | — | 无收益：8.4ms 是每帧固定开销，与像素量无关 |

启动形态很讲究（都实测过）：viewer 用 `env -i` 净化环境 + 输出重定向到文件
（`run_gsplat.sh --gui` 已封装）；交互 shell 的 bashrc 环境、ros2 launch 的管道
捕获、外部 xdotool 改窗口尺寸都会让它掉进 ~1s/帧的呈现路径。launch 的
`gui:=true` 目前窗口能起但帧率会劣化到 ~1 Hz（仿真本身仍 rtf 1.00）——要看窗口
就用 `run_gsplat.sh --gui`。另外 `motrixsim.run.render_loop` 每帧只把渲染耗时记入
物理累加器（步进耗时丢弃 → rtf 恒定 ~0.5），`sensor_server` 用自己的正确循环替代。

上游窗口 demo（独立仿真实例、键盘开狗）在宿主机仍可用：
`/home/dm/workspace/github/simulation/run_gs_demo.sh nav-church`（X11 + CUDA env
已封装在脚本里）。

## 布局

```
gsplat/
├── run_gsplat.sh        # 启动器：每组件一个 tmux 窗口（对标 gazebo/run_simulator.sh）
├── kill_gsplat.sh       # 全停 + 清 /dev/shm ring
├── gs_state.sh          # 对标 gazebo/dog_state.sh（真值从 ring v2 GT 通道读）
├── setup_env.sh         # 容器内一次性环境 bootstrap（宿主机执行）
├── launch/
│   └── gsplat_sim.launch.py  # 分机 sim 面独立 launch（对标 sim.launch.py）
├── server/
│   ├── sensor_server.py # 无窗口仿真服务（自包含：collect/policy/robot 已内联）
│   ├── gs_sensor_ring.py# 环形缓冲 v2（相机 + IMU + 真值）
│   └── gs_state_shm.py  # 机器人状态槽（viewer 用：sim_time + dof_pos @50Hz）
├── tools/
│   └── view_window.py   # 独立 viewer 窗口（读状态槽，与仿真进程解耦）
├── ros/
│   └── gs_ros_bridge.py # rclpy 桥（含 /sim/gt_pose；IMU frame_id 已修正）
├── robots/go2/
│   └── make_sensor_rig.py   # 生成 D435i rig MJCF（写到 GSPG 的 go2 目录、与
│                           #   go2_mjx.xml 同目录——网格按文件所在目录解析）
└── configs/             # church_go2.json / nav1_go2.json（出生点/朝向/资产路径）
```

## 与 gazebo 的差异（用的时候要知道）

- **真值**：无 `ign topic`；`gs_state.sh` 读 ring，桥发 `/sim/gt_pose`（50 Hz，
  MJCF world 系）。**IN/OUT 判定双轨**：SLAM odom 直判（精确——odom 就在地图系，
  需 `--slam`）+ GT `--yaw-deg -90` 旋转判（近似：VIO 去除的是估计出生航向
  ~94°≠配置 90°，10 m 处残差 ~1 m，栈不在时才用）。
- **步态**：RL 策略（onnx）而非 ros2_control IK 步态链；`/cmd_vel` 语义相同。
  没有楼梯/爬坡能力验证，场景=平地。
- **无 `--auto` 剧本场景**（scene_runner 是 gz 专属）； scripted 任务用 server 的
  `--drive-*`。
- **自遮挡剔除**：`--self-cull 0.3`（默认开）——per-link 机器人高斯比真实肢体
  胖，鼻尖相机会泡进自己躯干的高斯里（画面下半一大团失焦白雾）。它不是渲染
  或传输 bug。
- **渲染太干净**：无噪声/运动模糊/IR 质感，VIO 精度会虚高（既登记待办）。
- **DDS**：桥与栈必须同 CycloneDDS 且清空镜像烤入的 `CYCLONEDDS_URI`（Fast DDS
  下 783 KB color 图掉到 2.5 Hz）——run_gsplat.sh 已处理。

## 已知坑（bring-up 期 + 集成期实测，详见 docs/gsplat-sim-progress.md）

1. 起停顺序：**先 server（等 ring 出现）→ 再桥 → 再录制/栈**；写入方重启后
   ring seq 归零，读者靠 seq<last 重置（已实现）。
2. `img.data = bytes` 在 rclpy 是逐元素转换（mono 10.6 ms / rgb 33 ms）→ 桥用
   `frombytes` 批量拷（0.12 / 0.5 ms）。
3. GPU→CPU 先在 GPU 转 uint8 再拷（453→188 µs/张）。
4. 教堂过道沿 y 轴（-20…+20），出生四元数 (0,0,0.7071,0.7071) 朝 +Y；改
   `initial_qpos` 后必须 `forward_kinematic`（server 已内置）。
5. **cmd 文件必须原子写**（tmp + `os.replace`，桥已内置）：`write_text` 的
   O_TRUNC 空窗被 server 20 ms 轮询命中 → 空指令崩溃（闭环首分钟必现，
   手动发 cmd 永远踩不到）。
6. **GT 突发与轮询节奏**：渲染阻塞后物理追赶期 GT 按 sim 时间密集写槽，
   seq 门控的读者会合并样本（18 Hz）——GT 是状态量，桥用 50 Hz 定时器发
   当前槽内容（49.9 Hz，已内置）。
