# gsplat 仿真集成进度（3DGS / gs_playground）

活文档：集成、验收、踩坑按发现顺序追加（时间线见 git 历史）。目标形态 =
`gsplat/` 与 `gazebo/` 同级的完整仿真层（环境、传感器、控制接口），全部进程在
rig 容器 `tinynav` 内。本文件记录落地事实。

## 背景与来源

- 上游：`/home/dm/workspace/github/simulation/gs_playground`（外部 checkout，
  不改）。物理 MotrixSim（Rust，pypi.motphys.com 私有索引）+ 3DGS 渲染
  （gsplat 1.5.3，CUDA JIT）+ go2 RL 步行策略（onnx，CPU）。
- bring-up 期（宿主+容器分体形态）已验证：传感器面与 gz 逐字节
  一致、C++ 栈吃流在线建图（church_corridor_04 bag → 28 kf / 8.10 m）、
  `/cmd_vel` 反向闭环、CycloneDDS 大图丢包坑。这批接口代码（sensor_server /
  ring / 桥 / rig 生成器）当时以未提交状态散在 gs_playground demo 目录。
- 收进本仓 `gsplat/`（成为唯一归宿），ring 升 v2 加真值通道，
  全部进程入容器。

## 环境事实（容器 tinynav 实测）

| 项 | 值 |
|---|---|
| GPU | RTX 5070 Laptop 8 GB（容器直通，driver 610.43.02） |
| 容器自带 CUDA | 12.2（**不支持 sm_120**，gsplat JIT 不可用）→ 装 12.8 |
| `/opt/cuda-12.8` | 宿主 micromamba 前缀 docker cp 而来（1.3 G 自包含），`/opt/cuda-shim-gs` 做标准布局 shim |
| `/opt/venv_gs` | python 3.10（容器系统 python 同版本）：torch 2.7.0+cu128、gsplat 1.5.3、motrixsim-core 0.7.1.dev97295（pypi.motphys.com）、gaussian_renderer 0.2.0、onnxruntime 1.22.1、numpy 2.2.6 —— 与 ROS 栈 `/opt/venv` 完全隔离 |
| 镜像源 | 容器内 tuna 403；pypi.org / aliyun / motphys / download.pytorch.org 均通；uv 在 `/root/.local/bin`（0.7.3）；`UV_HTTP_TIMEOUT=600` 必需（nvidia wheel 数 GB，默认 30s 超时踩过） |
| gsplat kernel | `TORCH_CUDA_ARCH_LIST=12.0` JIT 编译一次入 `/root/.cache/torch_extensions`；**运行期仍需 `CUDA_HOME`+`PATH` 可见 nvcc**，否则 gsplat 判定无 toolkit 拒载已编译 kernel |
| 磁盘 | venv ~7 G + cuda 1.3 G，容器余 868 G |

## 进度

### P0–P3 落地（全链验收通过）

- [x] P0：`sim/` → `gazebo/` 改名（commit 0b2bf9a，43 文件纯 rename+引用；
      `/sim/` 话题名与 `tinynav_sim` tmux 名保留）。改名后 gz 冒烟：sensor 栈起、
      color 15 Hz、dog_state.sh 新路径读真值正常。
- [x] P1：容器环境。torch 走网络装太慢（download.pytorch.org ~20 MB/min），
      改为宿主 venv site-packages tar 流式拷入（7G，分钟级）+ 宿主 torch_extensions
      kernel 缓存直接复用（免 JIT）。渲染自检：教堂 3 相机 48-54 ms/帧、
      IMU 200.2 Hz、GT 50 Hz、61 帧/4s。
- [x] P2：代码入仓 `gsplat/`：
      - `server/gs_sensor_ring.py`：ring **v2** —— 头部 56/64 加 gt_seq/gt_time，
        尾部 64 B 真值槽 `[t, pos(3), quat(4)]`，server 50 Hz 写 base 位姿。
      - `server/sensor_server.py`：自包含（collect_*/Go2Robot/Policy 内联，
        甩掉 robot_locomotion→motrixsim.render 导入链）；`--duration 0` 常驻；
        rig 生成物放 GSPG go2 目录（网格按 MJCF 所在目录解析，仓库只留生成器）。
      - `ros/gs_ros_bridge.py`：+`/sim/gt_pose`（50 Hz 定时器，见坑 2）；修
        IMU frame_id（原误写 color_link）；cmd 文件原子写（见坑 3）。
      - `run_gsplat.sh` / `kill_gsplat.sh`（TERM→KILL+僵尸过滤）/ `gs_state.sh`
        （双判定，见坑 4）/ `configs/` / `robots/go2/`。

### 验收记录（容器内全链）

| 关卡 | 结果 |
|---|---|
| 传感器面 | color 15.8 Hz / infra1 15.0 Hz / IMU 桥侧 200+ Hz（CLI 慢消费约 147 Hz，栈实测正常）/ GT 49.9 Hz |
| `/sim/gt_pose` | 站立位姿 z=0.281、yaw≈83°（=教堂出生朝 +Y），内容正确 |
| `/cmd_vel` 闭环 | 4s 前进指令 → GT y: −0.003 → 1.658 m（真值沿过道前进） |
| C++ 栈吃流 | Baseline 0.0510 m、Initial yaw removed 93.93°、ISAM 持续优化 |
| 在线建图 | `/mapping/start` → 直行任务 12.00 m → `/mapping/stop`：**52 kf / 12.38 m / VLAD 52×24576 零重复行 / speed 中位 0.483 m/s / occupancy {0,1,2}**，52 MB v2 直出 |
| 回灌导航 | 图载入 reloc enabled；发目标 (10,0)（=出生直行 10 m，与建图同路）→ 规划 sel vx/omega、跟随器出 cmd_vel、狗走出 **9.89 m（差 11 cm）后驻停 77s+**；会话 reloc 15 hits（inlier 0.76–1.00） |
| gs_state | odom 精确判定 **IN**（9.89,−0.05 ∈ bbox x[0,12.32] y[−0.5,1.43]） |
| 稳定性 | 闭环持续 cmd_vel 下 server 90+s 无崩溃（坑 3 修复后）；kill 脚本收干净（GPU 回 14 MiB） |

## 已知坑（本阶段新增，都实测踩过）

1. **uv 默认 `UV_HTTP_TIMEOUT=30s`** 下载 nvidia 大 wheel 必超时——网络路线要
   600；本机最终走了 venv 拷贝路线（setup_env.sh 主路径）。
2. **GT 单槽 + 写方突发 = 读方合并**：server 渲染期阻塞 ~55 ms → 物理追赶期
   GT 按 sim 时间突发（~4 ms 间隔）→ seq 门控的"有新才发"读者只见到 18 Hz。
   GT 是状态量，桥改成 **50 Hz 定时器无条件发当前槽内容** → 49.9 Hz。
3. **cmd 文件截断-写入竞争（闭环首分钟必崩）**：桥 `write_text` 是 O_TRUNC 后写
   （非原子），闭环下 control 高频重写，server 20 ms 轮询命中空窗 → loadtxt
   返回 shape (0,) → 策略广播崩溃。修复双侧：桥 tmp+`os.replace` 原子替换；
   server 读侧 size<3 时保持上一条指令。bring-up 期手动发 cmd 永远踩不到，
   **闭环流量才暴露**。
4. **教堂地图系 vs MJCF 世界系差 ~90°**：VIO 去除的是"估计的出生航向"（93.93°
   ≠ 配置 90°），GT 旋转 `-90°` 后对比 bbox 在 10 m 处仍有 ~1 m 残差。
   `gs_state.sh --map-dir` 双判定：**SLAM odom 直判（精确，odom 就在地图系）**
   + GT `--yaw-deg` 旋转判（近似，栈不在时用）。
5. **僵尸进程骗存活检查**：容器 PID 1 不收割，tmux kill 后 tinynav_node 以 Z
   状态留在进程表——kill 脚本的 alive 检查要过滤 `^Z`（无 GPU/内存占用，只是
   进程表噪音）。
6. 桥在大栈订阅下 per-image 5.5 ms（pubcam 16.6 ms/tick，CLI 轻消费时 1.5-2.2
   ms）——RELIABLE 大图对 C++ 消费者的流控行为，15 Hz 预算内，未调优。

### 分机部署落地（gsplat_sim.launch.py + Orin 重刷后重建）

- `gsplat/launch/gsplat_sim.launch.py`：sim 面独立 launch（对标 sim.launch.py，
  DDS 接线/remote_planning/teleop/rviz 全部同构复用），sensor_server 由
  ExecuteProcess 拉起并自带 venv_gs + CUDA shim env——也治了 tmux 形态每个窗口
  手动重导环境的问题。Orin 侧 `orin_stack.launch.py` 零改动。
- **Orin 重刷后状态**（Ubuntu 22.04.5 / JetPack 6.2.3 / kernel 5.15.199-tegra，
  裸机 **Humble** desktop 已在）：本阶段补装 `ros-humble-rmw-cyclonedds-cpp`
  （注意包名是 `-cpp`，`-php` 不存在）、部署 `~/cyclonedds_orin.xml`（钉 l4tbr0）
  + `~/nav_env.sh` 一键环境。两侧同发行版（都是 Humble）后，当初为 Jazzy/
  Humble 类型不匹配选 CycloneDDS 的理由消失，但仍是指定配置（接口钉死、与
  生产一致）。工作区/docker 尚未在 Orin 重建——本验收用宿主机 ros2 CLI 做话题级
  验证。
- **验收（USB NCM 直连链路，话题级）**：
  | 检查 | 结果 |
  |---|---|
  | 发现 | Orin `ros2 topic list` 见全部 13 个仿真面话题（含 /cmd_vel、/sim/trajectory_path 反向订阅） |
  | 数据正向 | infra1 图像 15 Hz（min 62 / max 72 ms 稳定）、IMU 200 Hz、/clock、GT 50 Hz |
  | 感知出站 | x86 perception.launch.py 起后，`/camera/camera/slam/*` 8 话题 Orin 全可见，odometry_visual ~5 Hz、keyframe_image 过链 |
  | 反向 | Orin `ros2 topic pub /cmd_vel`（x=0.4 持续 6 s）→ 狗 GT y: 0 → 1.92 m |
- 收摊后 GPU 归零、ring 清理正常（kill_gsplat.sh）。

### 上游式可视化窗口（--gui）

- `sensor_server --window`：同进程内嵌 RenderApp（主视口=碰撞网格，左=机载相机
  +3DGS 全景），`run_gsplat.sh --gui` / `launch gui:=true` 接线，随仿真一起起。
- **容器 GL 现状**：无 NVIDIA GLX/Vulkan ICD，wgpu 走 Mesa llvmpipe（软件渲染
  ~24–57 ms/帧）；NVIDIA EGL 下 wgpu panic。`/root/.bashrc` 为 rviz/gz 烤了
  PRIME/EGL 三件套——server 在 window 模式主动 `os.environ.pop`，任何启动路径
  （docker exec / tmux 交互 shell / launch）都安全。
- **上游 render_loop 有 bug**：物理计时基准在步进循环之后才更新，每帧只把渲染
  耗时记入累加器 → 仿真恒定 rtf ≈ R/(R+S) ≈ 0.5（与面板/帧率无关，实测 0.45–0.49
  横跨四种配置）。已用自写循环替代（正确累加 + 帧率 pacing + 0.25 s 防螺旋封顶）。
- **实测预算**（这是取舍依据）：相机流 0.85 s/s（headless 已占 85%）+ 视口
  0.24–0.86 s/s（软件渲染）+ 面板 0.57 s/s@60fps（限流后≈0）。故 `--gui` 默认
  `--window-fps 15 --window-panels off`；`--cam-hz 8` 可回 rtf ~0.9。满血传感器 +
  流畅窗口需独立 viewer 进程（未做）。

### 可视化：独立 viewer 进程（替换内嵌窗口）

- **动机**：内嵌（`--window`）时 llvmpipe 视口与相机流抢主线程，rtf 掉到 0.45–0.68；
  视口尺寸不是杠杆（实测 1280×720 = 8.4ms/帧，320×180 = 10.0ms/帧——每帧固定
  开销，与像素量无关；`RenderApp` 也无尺寸参数，`_exper.render_app_with_batch_render`
  是批渲染专用，当 viewer 用报 "Could not get export buffer"；外部 xdotool 改尺寸
  会把呈现路径打到 1000ms/帧）。
- **实现**：`server/gs_state_shm.py`（160 B 状态槽：seq + sim_time + 19×f32 dof_pos，
  浮动基位姿在 dof_pos[0:7]，seq 后写防撕裂）+ `tools/view_window.py`（独立进程
  加载同一 MJCF、每帧 set_dof_pos + forward_kinematic + sync，鼠标轨道）。
- **实测**（church，含桥/控制/rviz）：独立 viewer → **仿真 rtf 0.91–0.92 + 窗口
  27–30 Hz**；内嵌 → rtf 0.68 + 15 Hz。**已替换为 `--gui` 默认**，内嵌保留为
  `--gui-embed` / `gui_embed:=true`。
- **启动形态是关键**（都实测过）：viewer 必须 `env -i` 净化 + 输出到文件——
  交互 shell 的 bashrc 环境、ros2 launch 的管道捕获、共享 session 都会让它掉进
  ~1s/帧的 wgpu 呈现路径（一旦进入不可恢复，杀 server 也不恢复）。run_gsplat.sh
  已封装正确形态；launch 的 `gui:=true` 目前窗口可起但会劣化到 ~1 Hz（仿真本身
  仍 rtf 1.00），要看窗口用 run_gsplat.sh --gui。
- 另修：上游 `motrixsim.run.render_loop` 物理累加器漏算步进耗时（rtf 恒 ~0.5），
  `sensor_server` 自写循环替代（正确累加 + 帧率 pacing + 0.25s 防螺旋封顶）。

### 默认无窗口 + 资产依赖显式化

- **MotrixSim viewer 默认关闭（`--gui` 才开）**，rviz 保持默认启动（与
  gazebo/run_simulator.sh 一致；`--no-rviz` 可关）。理由：viewer 走 llvmpipe，
  实测开它会把相机渲染从 52 ms/帧推到 75 ms/帧；`run_gsplat.sh` 默认整机（含
  rviz）实测 rtf **1.00**、桥 15.2/214.6/50 Hz。launch 侧 `gui` default false、
  `rviz` default false（分机站点不需要窗）。
- **资产依赖写入文档** `gsplat/README.md` §资产依赖与路径配置：GS_PLAYGROUND_ROOT
  的解析规则（configs 路径相对 `<root>/demo/navigation`）、逐项清单与体积、
  rig 必须生成在 go2 MJCF 同目录的原因。
- **新增 `gsplat/tools/check_assets.py`**：按 config 解析全部依赖 → 逐项 OK/MISSING
  + 体积，末尾打印打包给同事的 tar 命令与接收步骤（church+go2 = 156.5M / 6 项）。
  场景的正确依赖单位是**整个场景目录**（MJCF 的 `<include>` 与 meshes 按相对路径
  解析：church_collision.xml、nav_scene_1/meshes）。

### 资产包（交付给同事）

- `output/gsplat_assets.tgz`：141 MiB（解包 174.6 MiB），md5
  `fd0384dbca8450621f3768c12086b677`；内容 = church_scene + nav_scene_1 + go2
  plys(assets/go2) + go2 MJCF/assets + 生成的 rig xml + 策略 onnx + asound 配置。
- 两层验证：独立目录解包后 `check_assets.py --scene all` 全 **ALL PRESENT**；
  把 `GS_PLAYGROUND_ROOT` 指向解包目录真跑 `sensor_server --out`（766 万高斯、
  15.2 Hz 相机、200.2 Hz IMU、3 张 PNG）。
- 接收方流程（含 md5 校验、路径设置、冒烟命令）写在 `gsplat/README.md`
  §资产依赖与路径配置 →「现成的资产包」。

## 观察待跟进（不阻塞使用）

- 到达行为：目标差 11 cm 驻停后 planning 仍 sel vx=0.20、跟随器仍发 0.2 m/s，
  狗不动（策略低速死区或到达逻辑）——导航调参细节，非集成缺陷。
- 渲染太干净（无噪声/模糊/IR）→ VIO 虚高（既登记待办）。

## 范围外（明确不做，需要时另立项）

DISCOVERSE 仿真；图像噪声/运动模糊/IR 质感（VIO 虚高问题）；848×480 真机
分辨率参数化；gs_gt_reloc 移植（`--map` 模式 C++ 栈自带 reloc，暂不需要 gz
式真值 reloc）；分机 launch（`gsplat.launch.py`）。
