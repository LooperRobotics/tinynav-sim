# sim 与建图运行手册

日常操作的启动顺序、建图流程和现场自检。排错向的历史沉淀见
`docs/migration-progress.md`；本文件只写"现在怎么跑"。

## 1. 环境与编译

一切在容器里跑，**别在宿主机编译**。日常用常驻 rig 容器 `tinynav`
（仓库挂载在 `/workspace/dm/tinynav-sim`，非 `/ws`）：

```bash
docker start tinynav && docker exec -it tinynav bash
# 容器内：
cd /workspace/dm/tinynav-sim
source /opt/ros/humble/setup.bash
colcon build --packages-select tinynav_cpp
```

改过代码或 config 后必须重编——launch 读的是 `install/` 下的拷贝。
换过容器实例后也必须重编（CMake 缓存里有旧的绝对路径）。

容器内手动跑 ros2 CLI / 探针前，先覆盖两个 DDS 环境变量（镜像 ENV 烤死的
`CYCLONEDDS_URI` 指向不存在的文件，所有 Cyclone 节点会建域即死）：

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=""
```

`run_simulator.sh` 起的窗口已自动带这两个变量；只有手开终端需要自己 export。

## 2. 启动仿真和程序（单机，日常默认）

一条命令起齐 gz + C++ 导航栈 + rviz，每组件一个 tmux 窗口：

```bash
# 带地图导航（C++ 栈，重定位在环）
bash sim/run_simulator.sh --stack cpp --map --map-dir output/map_v2 \
    --world sim/worlds/yard.sdf

# 不带地图（纯感知+规划自由跑）
bash sim/run_simulator.sh --stack full --robot go2

# 看各窗口输出 / 收摊
tmux attach -t tinynav_sim        # 窗口: gz gui robot bridge caminfo control teleop cpp rviz
bash sim/kill_sim.sh              # 启动器每次自己也会先跑它
```

参数速查：

| 参数 | 说明 |
|---|---|
| `--stack full\|sensor\|cpp` | full=python 全栈；cpp=单进程 C++ 栈（tinynav.launch.py）；sensor=只传感器面 |
| `--map` + `--map-dir <dir>` | 加载 map format v2 目录（`tools/export_map_v2.py` 产物），启用 C++ 重定位 |
| `--world <sdf>` | 必须给 **SDF 路径**（如 `sim/worlds/yard.sdf`），不是名字；empty 无纹理，**reloc 验证用 yard** |
| `--auto <scene>` | 剧本场景（与 `--stack cpp` 互斥） |
| `--db <path>` | 各窗口 TINYNAV_DB_PATH |

启动顺序（脚本已排好）：gz（时钟源）→ 传感器面 → 栈。`--stack cpp` 时
`sim_gt_reloc` 真值重定位自动关闭，定位权威 = C++ mapping 组件本身。

发导航目标（gz 相机 15Hz、世界系坐标）：

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=""
source install/setup.bash
ros2 topic pub -w 1 -r 5 -t 8 /control/target_pose nav_msgs/msg/Odometry \
  "{header: {frame_id: world}, pose: {pose: {position: {x: 5.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}}"
```

## 3. 开工前的状态确认（纪律）

```bash
bash sim/dog_state.sh --slam --map-dir output/map_v2
```

输出 gz 真值位姿+yaw、SLAM odom、与建图轨迹包围盒的 IN/OUT 判定。
**机器人不在建图轨迹上 / 朝向不符时，reloc 连败是预期行为，不是 bug**——
先回轨迹再排查。SLAM odom 与 gz 的 yaw 差恒定 ~90°（安装偏置），各自对比、勿互比。

链路数据面探针（确认某话题有没有在发）：

```bash
python3 tools/probes/probe_first_msg.py /camera/camera/slam/odometry_visual Odometry
```

## 4. 建图

### 4.1 sim 里的建图 → 给 C++ 栈用（当前主要工作流）

C++ 栈只吃 map format v2；图的**生产**仍走 python 工具链，两步：

```bash
# 终端 1：起活栈（无图模式即可建图）
bash sim/run_simulator.sh --stack full

# 终端 2：python BuildMapNode 直接吃实时 /slam/keyframe_* 流（跳过录包）
python3 tools/build_map_live.py --map_save_path output/map_build_test
#   采完 Ctrl+C —— save_mapping() 收尾：位姿图 + VLAD 训练 + occupancy 烘焙

# 导出成 v2（u16 毫米深度 + f32 特征/VLAD，lazy 友好的列式布局）
python3 tools/export_map_v2.py output/map_build_test output/map_v2
```

之后 `--stack cpp --map --map-dir output/map_v2` 即可加载导航。
**改过地图格式后必须用 numpy 回读验证**（C++ 解析器宽容，别只信它）：

```bash
python3 -c "import numpy as np; print(np.load('output/map_v2/depth_images.npy', mmap_mode='r').shape)"
```

生成压力测试夹具（地图格式回归测试用，~2.6G，进 `fixtures/` 不进 git）：

```bash
python3 tools/make_stress_map_v2.py            # 默认 4000 帧
colcon build ... && ./build/tinynav_cpp/tinynav_core_test   # 72 用例应全绿
```

### 4.2 C++ 栈的自身帧落盘（导航时自动发生）

导航/建图过程中，mapping 组件把自身关键帧写入
`live_capture_dir`（默认 **`fixtures/nav_temp_v2`**，gitignored，每次会话自动
清空）——深度 u16、特征列式追加，回环按候选帧读回，进程内存不随任务时长增长。
组件退出时自动 finalize 成合法 npy（+poses+VLAD 即是 v2 地图子集，
这是后续在线建图的落点）。

### 4.3 狗上生产建图（现状）

狗上仍是 python 栈的 web 流程：`/bag/start` → 途中 `/bag/poi-marks` 打点 →
`/bag/stop` → `/map/build`（bag 回放建图）→ 建完目录在 `tinynav_db/maps/`。
给 C++ 栈用前同样要 `export_map_v2.py` 导出。C++ 在线建图尚未落地，
落地后此节将被"统一会话采集 + 一次 finalize"取代。

## 5. 重定位排查工具

- **失败现场 dump**：启动栈前 `export TINYNAV_RELOC_DUMP_DIR=<dir>`，每个被拒
  候选（LightGlue <50 对）把完整 LG 输入写入 `<dir>/<live_ts>_<cand_ts>/`，
  用 `tools/probes/probe_lg <dir>` 回放——回放匹配数必须等于 meta.json 的
  match_count（与组件逐位对齐）。
- **成功判据**：`/map/relocalization` 有发布且位姿与里程计吻合；
  日志关键字 `map v2 loaded: ... keyframe relocalization enabled`。
  注意机器人在轨迹外时连败是正确行为（见第 3 节）。

## 6. 分机部署（备用）

x86 跑 sim+perception、Orin 跑 bridge+栈的拓扑（`sim.launch.py` +
`perception.launch.py` / `orin_stack.launch.py`，CycloneDDS 钉 USB 网卡），
操作步骤见 `sim/README.md`。日常单机开发用不到。
