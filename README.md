# tinynav-sim

tinynav 热路径节点（perception / imu_propagator / mapping / planning）的 C++ 重写工作区：
单进程 rclcpp 组件 + 进程内零拷贝，以两种仿真（gzsim、3DGS）和 bag 回放作为端到端测试台。
`reference/` 是只读 Python 快照，是移植的 spec（冲突时 Python 语义优先）。

## 1. 目录结构（每个子文件夹干什么）

```
gazebo/              gzsim 仿真层：worlds（empty/yard/depot/factory/stairs）、robots
                     （go2 URDF+步态链 / lekiwi）、scene（剧本场景/真值 reloc/发目标）、
                     run_simulator.sh（tmux 一键编排）——详见 gazebo/README.md
gsplat/              3DGS 仿真层（MotrixSim 物理 + gsplat 渲染 + RL 步行策略），传感器面
                     与 gazebo 逐字节一致；run_gsplat.sh / gs_state.sh / setup_env.sh
                     ——详见 gsplat/README.md
reference/tinynav/   Python 参考快照（core/*.py = 可执行 spec；platforms/ = 实机与仿真的
                     底盘适配节点）。只读，禁止修改
src/tinynav_cpp/     C++ 包本体：
  src/core/          纯数学（SE3、PnP、IMU 积分）
  src/kernels/       raycast / 位姿图 / BA 内核（原 pybind 下沉）
  src/planning/      占据栅格、ESDF、轨迹库（6 个 njit 函数的移植）
  src/mapping/       VLAD、fusion window、A*、路径先验、live capture
  src/trt/           TensorRT C++ 封装（8 引擎 + CUDA Graph）
  src/components/    四个 rclcpp 组件 + main.cpp 单进程宿主
tools/               侧工具：build_map_live.py（python 在线建图）、export_map_v2.py
                     （v1→v2 导出）、grab_topic.py（话题抓取）、make_stress_map_v2.py
                     （压力测试夹具）、probes/（TRT 对拍探针）
docs/               设计/操作文档（见文末索引）
fixtures/            对拍黄金数据与运行期产物（gitignored，永不进 git）
output/  logs/       地图/数据库与各窗口日志（gitignored）
```

## 2. 环境配置（相比 tinynav 仓库要多装什么）

基础环境与 tinynav 仓库**完全同一套容器**（ROS2 Humble + TRT 10 + GTSAM + venv），
本仓库不引入新的 ROS 依赖。常用两种容器（二选一，别混用）：

- **常驻 rig 容器**（日常验证用）：`docker start tinynav && docker exec -it tinynav bash`，
  镜像 `tinynav-runtime:x86_64`，仓库挂在 `/workspace/dm/tinynav-sim`；
- **一次性 dev 容器**（编译用）：`docker run --rm --gpus all --network host -v "$PWD":/ws
  -w /ws uniflexai/tinynav:latest bash`。

在此之上，本仓库额外需要三样（按需）：

| 需要什么 | 干什么用 | 怎么装 |
|---|---|---|
| colcon 构建 | C++ 栈（`--stack cpp`） | `source /opt/ros/humble/setup.bash && colcon build --packages-select tinynav_cpp`（**切换容器后必须重跑**，CMake 缓存带旧绝对路径） |
| `pynput` | 键盘遥操窗口（可选） | 容器内 `pip install pynput`，缺了只跳过该窗口 |
| 3DGS 仿真全家桶 | `gsplat/` 仿真器 | **一键**：宿主机 `bash gsplat/setup_env.sh`（见下） |

3DGS 一键脚本会装齐：`/opt/venv_gs`（torch 2.7.0+cu128 / gsplat / motrixsim 私有索引 /
onnxruntime，与 ROS venv 完全隔离）+ `/opt/cuda-12.8` nvcc（容器自带 12.2 不支持
RTX 5070 的 sm_120，gsplat JIT 需要 ≥12.8）+ kernel 预编译 + 渲染自检。回滚 = 删
`/opt/venv_gs`、`/opt/cuda-12.8`、`/opt/cuda-shim-gs` 三个目录。**场景资产**（教堂
ply、go2 高斯、策略 onnx）不进 git，需外部 gs_playground checkout，默认路径
`/workspace/github/simulation/gs_playground`（`GS_PLAYGROUND_ROOT` 可覆盖）。
详细过程与坑见 `docs/gsplat-sim-progress.md`。

## 3. 仿真模式

两个仿真器 × 三种栈形态，外加 bag 回放（无仿真器）。都在容器里跑。

| 模式 | gazebo（gzsim） | gsplat（3DGS 照片级） |
|---|---|---|
| 纯仿真面 `--stack sensor` | gz + 桥 + perception（pilot 拥有其余时的形态） | server + 桥（pilot/其他消费者拥有栈） |
| python 参考栈 `--stack full` | gz + reference 感知/规划（+`--map` 起 python map_node） | 3DGS + reference 感知/规划 |
| C++ 产品栈 `--stack cpp` | gz + 单进程 C++ 四组件 | 3DGS + 单进程 C++ 四组件（**日常验证主力**） |

```bash
# gazebo（容器内，cd /workspace/dm/tinynav-sim）
bash gazebo/run_simulator.sh                          # 默认 --stack full --robot go2 --world empty
bash gazebo/run_simulator.sh --stack cpp --map --map-dir output/map_v2 --world gazebo/worlds/yard.sdf
bash gazebo/run_simulator.sh --auto l_corridor        # 剧本场景（gz 专属）

# gsplat
bash gsplat/run_gsplat.sh --stack cpp                 # 建图模式（无图）
bash gsplat/run_gsplat.sh --stack cpp --map --map-dir output/map_gs_church
bash gsplat/run_gsplat.sh --stack cpp -- --drive-start 25 --drive-distance 12  # 剧本直行

# bag 回放（无仿真器；/clock 由 bag 提供，栈需 use_sim_time）
ros2 launch tinynav_cpp tinynav.launch.py params_file:=<yaml>   # yaml 设 use_sim_time:true
ros2 bag play <bag> --clock
```

发目标（cpp 栈直接发 `/control/target_pose`；gsplat 教堂图系 = 出生直行方向）：

```bash
ros2 topic pub -w 1 -r 5 -t 5 /control/target_pose nav_msgs/msg/Odometry \
  "{header: {frame_id: world}, pose: {pose: {position: {x: 10.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}}"
```

分机部署（x86 跑仿真+感知、Orin 跑其余三件）用 launch 形态：gazebo 见
`gazebo/README.md`，gsplat 见 `gsplat/README.md` 分机部署节（Orin 侧零改动）。

## 4. 常用功能

- **在线建图（C++，推荐）**：`--stack cpp` 不带 `--map` = 建图模式；开狗走图，服务
  `/mapping/start` / `/mapping/stop`（std_srvs/Trigger）直接产出 map format v2
  （采集 npy 即图文件，零转换）。验收数据与细节见 `docs/sim-mapping-runbook.md` §4.4。
- **python 建图路线（备用/对照组）**：`tools/build_map_live.py` 吃实时 keyframe 流，
  `tools/export_map_v2.py` 导 v2 —— runbook §4.1/§4.3。
- **带图导航 + 重定位**：`--map --map-dir <v2 目录>`；发目标走上面的 pub。reloc 排查
  工具（失败现场 dump + LightGlue 回放对拍）见 runbook §5。
- **开工纪律**：每次新操作前先确认狗位姿——gz 用 `bash gazebo/dog_state.sh --slam
  --map-dir <图>`，gsplat 用 `bash gsplat/gs_state.sh --slam --map-dir <图>`；轨迹外
  的 reloc 连败是预期行为，先重启再排查。
- **对拍/单测**：`colcon build` 后 `./build/tinynav_cpp/tinynav_core_test`（72 用例）；
  TRT 探针 `tools/probes/`；压力地图夹具 `tools/make_stress_map_v2.py`。

## 5. 文档索引

| 文档 | 内容 |
|---|---|
| `gazebo/README.md` | gz 仿真层：两种启动方式、分机部署、DDS、楼梯世界 |
| `gsplat/README.md` | 3DGS 仿真层：架构、bootstrap、用法、与 gazebo 的差异 |
| `docs/sim-mapping-runbook.md` | 建图操作手册（§4.4 C++ 在线建图、§5 reloc 排查） |
| `docs/gsplat-sim-progress.md` | gsplat 集成进度、验收记录、坑 |
| `docs/stairs-gait.md` | 楼梯世界与盲爬步态适配（暂停存档） |
| `docs/migration-progress.md` | 移植进度总账（历史） |
