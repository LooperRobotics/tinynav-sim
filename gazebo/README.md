# gazebo/ — gzsim 仿真层

从老 tinynav 仓库的 `tool/simulator/` 搬出，路径已适配本仓库布局。对应关系：

| 老位置（tinynav 仓库）            | 新位置（tinynav-sim）      |
| --------------------------------- | -------------------------- |
| `tool/simulator/worlds/`          | `gazebo/worlds/`              |
| `tool/simulator/robots/{go2,lekiwi}/` | `gazebo/robots/{go2,lekiwi}/` |
| `tool/simulator/gazebo_scene/`    | `gazebo/scene/`               |
| `tool/simulator/kill_sim.sh`      | `gazebo/kill_sim.sh`          |
| `scripts/run_simulator.sh`        | `gazebo/run_simulator.sh`     |
| `tinynav/core/*.py`（uv run 跑）  | `reference/tinynav/core/*.py`（直接 `python3` 跑） |
| `tinynav/platforms/{simulator_control,keyboard_teleop.py}` | `reference/tinynav/platforms/` |

## 两种启动方式（run_simulator.sh 与 sim.launch.py 的分工）

- **`gazebo/run_simulator.sh`**（tmux 9 窗口）：单机开发全家桶——sim + 栈 +
  rviz 一个命令起齐，每窗口可看实时输出、`tee logs/*.log`。调试日常用它。
- **`gazebo/launch/sim.launch.py`**（ros2 launch）：sim 层独立产品面——只含
  gz server/gui、spawn、gz bridge、camera_info、simulator_control（teleop
  可选），**不含任何导航栈**。分机部署（x86 跑 sim+perception，Orin 跑
  bridge+三件）用它与 `tinynav_cpp/launch/{perception,orin_stack}.launch.py`
  组合。

  **DDS**：两边默认 CycloneDDS（`dds:=cyclone`；launch 内置 export
  `RMW_IMPLEMENTATION` + `CYCLONEDDS_URI=file://.../src/tinynav_cpp/config/
  cyclonedds_x86.xml`，配置钉在 USB 链路网卡上）。两个站点 ROS 发行版不同
  （x86 Humble / Orin Jazzy），不能共用 Fast DDS：Jazzy 默认开启 type
  namespacing，类型标识与 Humble 永远对不上（rmw_fastrtps#797 还能把
  Humble 侧 OOM）。`dds:=fastdds` 保留旧 UDPv4 + discovery server 接线，
  仅用于单发行版环境。

  ```bash
  # ---- x86 站点（rig 容器 tinynav，仓库挂 /workspace/dm/tinynav-sim）----
  docker start tinynav && docker exec -it tinynav bash

  # 终端 1：sim 层（/clock 归它管）。remote_planning:=true 把跟随器重映射
  #        到 /sim/trajectory_path，接 Orin 桥过来的轨迹；gui/rviz 默认起。
  cd /workspace/dm/tinynav-sim && source /opt/ros/humble/setup.bash
  ros2 launch gazebo/launch/sim.launch.py world:=gazebo/worlds/yard.sdf robot:=go2 \
      remote_planning:=true

  # 终端 2：perception（/slam/* 重映射进 camera-box 命名空间出站）
  source install/setup.bash
  ros2 launch tinynav_cpp perception.launch.py

  # ---- Orin 站点（裸机 Jazzy，nvidia@192.168.55.1）----
  sshpass -p nvidia ssh nvidia@192.168.55.1
  source /home/nvidia/workspace/dm/nav_env.sh   # Jazzy + 工作区 + cyclonedds_orin.xml + DB
  ros2 launch tinynav_cpp orin_stack.launch.py map_path:=/home/nvidia/workspace/dm/map_v2
  ```

  启动顺序：sim（时钟源）→ perception → Orin 栈（use_sim_time 吃跨链路
  /clock）。改过代码或 config 后先 `colcon build --packages-select
  tinynav_cpp`——launch 读的是 `install/` 下的拷贝，Orin 侧同理。想要
  per-node 日志就在两侧 export `TINYNAV_DB_PATH`（Orin 的 nav_env.sh 已带）。
  收摊：x86 `bash gazebo/kill_sim.sh`（单独一条跑）；Orin
  `pkill -f "[o]rin_stack"`（方括号防 pkill 匹配到自己的命令行）。

  起来后确认：x86 `bash gazebo/dog_state.sh --slam`（gz 真值 + yaw + 是否在
  建图轨迹包围盒内）；链路用数据面探针
  `python3 tools/probes/probe_first_msg.py /camera/camera/slam/odometry_visual
  Odometry`。Cyclone 模式下 ros2 CLI 正常可用（不再是 discovery-server
  全盲状态）；但容器里手动跑 CLI/探针要先手动 export 与 launch 相同的两个
  DDS 变量——镜像 ENV 烤进去的 CYCLONEDDS_URI 指向不存在的文件，必须覆盖。
  两容器演练全链已通：四话题跨机 + 行驶验收 + reloc 位姿 3cm；
  真机 Orin 同链路 C3 行驶验收通过（当时裸机 Jazzy；Orin 已重刷
  Ubuntu 22.04 + Humble，两侧同发行版，CycloneDDS 仍是指定配置）。

## 布局

```
gazebo/
├── run_simulator.sh   # 启动器：每组件一个 tmux 窗口（单机开发用）
├── launch/sim.launch.py  # sim 层独立 launch（分机部署；dds:=cyclone 默认，DDS 环境内置）
├── fastdds_udp.xml    # （备用）FastDDS 2.6 XML：UDPv4-only + 白名单
├── kill_sim.sh        # 全停（启动器每次先跑它，保证不串上一次）
├── worlds/            # empty / yard / depot / factory / stairs .sdf —— 纯环境，不含机器人
├── robots/            # go2（URDF + ros2_control + 步态链）、lekiwi（SDF 圆柱）
└── scene/             # 原 gazebo_scene：scene_runner / sim_gt_reloc /
                       # camera_info_publisher / gen_textures / config / models
```

> fastdds_udp.xml 曾在两容器演练中试过（transport_descriptors + 白名单写法）。
> 分机链路的最终方案是两边 CycloneDDS（`src/tinynav_cpp/config/
> cyclonedds_{x86,orin}.xml`：钉 USB 网卡 + 组播，无 server 进程）；此文件
> 仅在 `dds:=fastdds`（`FASTDDS_BUILTIN_TRANSPORTS=UDPv4` +
> `ROS_DISCOVERY_SERVER`）的单发行版场景下作为白名单/缓冲区调优起点。
> 注意 Humble 的 Fast DDS 2.6 XML 解析器不吃 xmlns。

## 怎么跑

全部在容器里跑（镜像 `uniflexai/tinynav:latest`），仓库挂到 `/ws`：

```bash
docker run --rm --gpus all --network host -it \
  -v /home/dm/workspace/dm/tinynav-sim:/ws -w /ws \
  uniflexai/tinynav:latest bash

# 容器内：
bash gazebo/run_simulator.sh                    # 默认 --stack full --robot go2 --world empty
bash gazebo/run_simulator.sh --robot lekiwi --world yard
bash gazebo/run_simulator.sh --auto l_corridor  # 剧本场景
bash gazebo/run_simulator.sh --stack sensor     # 只起传感器面（pilot 拥有其余部分）
tmux attach -t tinynav_sim                   # 看各窗口
```

日常更常用的是常驻 rig 容器 `tinynav`（`docker start tinynav && docker exec
-it tinynav bash`），仓库挂载在 `/workspace/dm/tinynav-sim`（不是 /ws）；
上面 `docker run --rm` 是一次性临时容器（/ws）。两条路径二选一，别混用。

PYTHONPATH 不用手配：脚本头部已 `export PYTHONPATH="$WS_ROOT/reference:$PYTHONPATH"`，
镜像 ENV 里本来就有 `/opt/venv/lib/python3.10/site-packages` + `/3rdparty/gtsam/build/python`
（注意：venv 在 `/opt/venv`，不是老文档说的 `/tinynav/.venv`）。`reference/tinynav/`
故意**没有** `__init__.py`：保持 namespace package，`tinynav.core` 解析到 reference 快照，
而 `tinynav.tinynav_cpp_bind`（编译好的 .so）仍从镜像 site-packages 解析——加了
`__init__.py` 反而把后者挡住。

已知缺口：镜像没装 `pynput`（老流程靠 `uv run` 从 uv.lock 同步），键盘 teleop 窗口
会被跳过并打 WARN；需要时容器里 `pip install pynput`。

## --stack cpp

perception / planning / map 三个 Python 窗口换成一个窗口跑单进程 C++ 栈。
先构建（仓库根已按 colcon 工作区组织）：

```bash
docker run --rm --gpus all --network host -v "$PWD":/ws -w /ws \
  uniflexai/tinynav:latest bash -c \
  'source /opt/ros/humble/setup.bash && colcon build --packages-select tinynav_cpp'

bash gazebo/run_simulator.sh --stack cpp     # 或 TINYNAV_STACK=cpp
```

cpp 栈直接拥有 `/slam/odometry_visual`：没有 raw 流 remap，因此没有
sim_gt_reloc，`--map` / `--auto` 与 `--stack cpp` 互斥。发目标用
`bash gazebo/scene/pub_target.sh`（直接发 `/control/target_pose`）。

## 贴图路径（model://）

世界 SDF 里的贴图引用是 `model://textures/<name>.png`，由启动器导出的
`IGN_GAZEBO_RESOURCE_PATH=$SIM_ROOT/scene/models` 解析（空 URI 解析失败时 ign 会打
`Unable to find file with URI`，脚本路径内已验证）。不依赖镜像里 `/tinynav` 的任何文件。

## stairs 世界（楼梯）

`gazebo/worlds/stairs.sdf`：两段楼梯 + 中间转向平台（L 形折返）。几何由
`gazebo/worlds/gen_stairs_sdf.py` 生成（改 RISE/TREAD/N 后重跑覆盖），当前为
**常用室内楼梯尺寸：踢面 0.15 m / 踏面 0.28 m / 宽 1.4 m，每段 8 级
（第一段顶 1.2 m，第二段顶 2.4 m）**，实心到地台阶 + STAIRS 混凝土踏面贴图
（gen_textures 生成，踢沿暗带+白线供 SLAM 特征）+ 低侧护栏 + 箱墙远场锚点。
go2 出生点用全局默认 (0,0,0) 朝 +x，第一段楼梯起点 x=2.0。

**步态现状（实测，非结论性的运控验收）**：sim 的 go2 是 IK
treadmill trot（skating 步态，落足高度取体相对值、无地形感知）——8 cm 踢面
单体能上（z_leg_lift 0.17 + 体高 0.28 时）但第二级起连续上不去；4 cm 踢面
驱动能穿过梯段但躯体高度不升。**要爬标准 15 cm 踢面需要真改步态**（落足点
随支撑面抬升 / 躯干高度与俯仰自适应），不是调常数能解决的；真机运控是宇树
自带栈，不受此影响。楼梯世界的当前价值 = 建图/climb 标签/导航链路验证
（path_climb 的 MIN_RISE=0.12 阈值对 0.15 m 踢面正常打标）。

**路线 A 盲爬适配：已暂停、步态修改已回退**。首轮实现在平地
回归中翻倒，根因已定位（反射被正常摆动滞后误触发 + joint_states 乱序 +
限速单位错），详见 `docs/stairs-gait.md` 第 4 节；适配版代码存档在
`gazebo/robots/go2/stair_adapt_wip/`。当前 `go2_controller.py` 为原版，平地
行为已验证正常。楼梯世界（15cm 标准踢面）保留可用，重启 sim 生效。

## 其他

- 日志在 `$WS_ROOT/logs/`；`--map` 的地图/数据库在 `$WS_ROOT/output/`（gitignored）。
- rviz 配置仍用镜像里的 `/tinynav/docs/vis.rviz`。
- factory 世界的 `model://factory_01` 厂房模型在仓库外，用 `FACTORY_MODEL_ROOT`
  环境变量指到包含 `gz_model/` 的目录。
- `--db <path>` 导出 `TINYNAV_DB_PATH` 给所有窗口（logsetup 的数据根）。
