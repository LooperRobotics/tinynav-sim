# 迁移进度与待办计划

# 迁移进度与待办计划

> 本文档随迁移滚动更新。最近更新：2026-09-20（分机部署架构落地：组件子集 +
> looper_bridge 中继 + 三 launch；两容器演练 A 侧数据面已验证，B 侧待验）

## 范围与决策（已与负责人确认）

- 目标：把 tinynav 热路径节点（perception / map / planning / imu_propagator）
  重写为 C++ rclcpp 组件，**单进程、进程内零拷贝通信**；先纯替换、不加新功能。
- 工作区：`/home/dm/workspace/dm/tinynav-sim`（独立仓库，与 tinynav-pilot 无关）。
- 已拍板：① reference 用纯快照（只拷 core py + cpp 内核），不做 subtree；
  ② fixtures 不进 git；③ 推进顺序 = planning 全文件 → 编译+节点启动 →
  git commit → 逐节点 Python 对拍 → 端到端仿真。

## 总体状态

| 模块 | 状态 | 验证 |
|---|---|---|
| 仓库骨架（README/AGENTS/.gitignore/包骨架/launch/config） | ✅ | — |
| `reference/` Python 快照（core/*.py + cpp/*.cpp + platforms 两个文件） | ✅ | import 实测通过 |
| `sim/` 仿真资产 + run_simulator.sh 路径适配 + `--stack cpp` 分支 | ✅ | 容器内 gz 冒烟通过（empty.sdf 加载、话题列表正常） |
| core math 库（math/imu/robot_specs） | ✅ | gtest + Python 对拍 |
| mapping 库（vlad/fusion_window/path_prior/astar/semantic_retrieval + pose_graph/BA kernels） | ✅ | gtest + pose_graph_solve 对拍（vs 镜像 pybind .so） |
| trt 封装（TRTBase + 8 模型，懒加载降级） | ✅ | gtest，无 GPU 时优雅降级；gzsim 实景推理验证 |
| **planning 库（6 个 njit 热点 + 栅格/ESDF/EDT + raycast kernel + DWA）** | ✅ | gtest + Python 对拍 10 组场景全绿 |
| **4 个 components + main.cpp 工厂** | ✅ | 单进程 4 组件 launch 冒烟，39 话题 |
| **容器内 colcon build → install** | ✅ | gtest 57/57（含 alignment） |
| **节点启动冒烟** | ✅ | 4 组件 up，intra-process on，无引擎存活 |
| **单元测试对齐 Python 输入输出** | ✅ | test_alignment_planning + test_alignment_core，对拍中修复 2 个移植 bug |
| **端到端仿真验证** | ✅（见镜像缺口） | gzsim yard + cpp 栈：深度→VIO→栅格→规划→cmd_vel 全链跑通 |
| **M2 map format v2**（导出器 + C++ 读取器 + reloc 接线 + --map） | ✅ | 合成图 round-trip 对拍；92kf 实图加载/行驶；reloc 匹配数问题已解决（img_shape 语义，见"已解决"节）|
| **perception v2 GTSAM 因子图**（CombinedImuFactor + SmartStereo + LM） | ✅ | 对拍误差与 python 完全一致（位姿 1e-8）；rig 实跑 VIO failed=0；CPU +23pp |
| **开销对比**（v1/v2/python，栈口径） | ✅ | yard 同协议采样；python 131.4% vs cpp 38.5/61.8% |

## 各模块要点（组件层照此消费）

### core math（`include/tinynav_cpp/core/`）
- pose7 约定 = `[x, y, z, qx, qy, qz, qw]`（已写进 math.hpp 注释）
- `estimate_pose`（PnP，对应 cv2.solvePnP）、`rerank_by_pnp_inliers`、`depth_to_cloud`、
  `process_keypoints`、`UnionFind`、`integrate`（IMU 预积分）、`robot_config(name)`/
  `robot_config_from_env()`
- 取舍：Python float32 内部统一为 double；estimate_pose 的 lru_cache 未移植

### mapping（`include/tinynav_cpp/mapping/`、`include/tinynav_cpp/kernels/`）
- VLAD streaming 训练/计算、FusionWindow（N=5 构造参数，env 由组件层读）、
  PathSpeed/PathClimb 索引（KdTree 为暴力实现，接口已留好）、A*（Grid3 + 4 个函数）、
  `pose_graph_solve`（ceres 实现）、`ba_solve`
- **map_node 契约已调研完毕**（话题清单 / QoS / keyframe_mapping + 2 个 timer 的状态机
  描述在 wave-1 汇报中，组件层照抄）：keyframe 三话题 TimeSynchronizer(10) 精确同步、
  发布 ~12 个话题、nav_target_timer 2Hz、tick_map_priors 2Hz、融合窗口 5 条约束、
  到达判定 0.5m/2 tick
- 取舍：bake() 未移植（读 pickle，留给 Python 离线工具链）；cKDTree→暴力精确解

### trt（`include/tinynav_cpp/trt/`）
- TRTBase：懒加载 + page-locked buffer + CUDA Graph（捕获失败降级直接 enqueueV3）；
  8 模型 infer 签名见 models.hpp；任何加载失败标记 unavailable，进程不崩
- SigLIP 文本 tokenizer 可插拔：`SiglipTokenizer` 接口 / 缺省 unavailable /
  `PrecomputedTextEncoder`（读离线预算的 .f32，适合 POI 集合固定的场景）
- 注意：`include/cuda_runtime_api.h` 是一个转发 shim（NvInfer.h 硬包含 CUDA 头，
  语法检查 -I 集合里没有 CUDA include 目录；colcon 构建时优先转发真头）

## 已知问题 / 风险清单

1. ~~planning 库缺失~~（已解决：`src/planning/` + `edt` + `kernels/raycast.cpp` + `test_planning.cpp` 全部落位）。
2. ~~CMakeLists 需要补 ceres~~（已解决：`find_package(Ceres REQUIRED)` 已链；注意 Humble 的
   `ament_target_dependencies` 不支持 PUBLIC/PRIVATE 关键字，target_link_libraries 需全 plain）。
3. **main.cpp 工厂契约**：4 个工厂函数已在各 `*_component.cpp` 末尾定义，
   返回 `std::shared_ptr<rclcpp::Node>`（具体类在 main.cpp 不完整，转换发生在 .so 内）。
4. **GTSAM 缺口**：镜像无 libgtsam-dev，perception v1 用 PnP + IMU 传播，
   因子图细化藏在 `GtsamRefine` 接口后面（默认 no-op），待 docker 层加 GTSAM 后补。
5. **map format v2 未做**：mapping 组件无法读 poses.npy（pickled）与 TinyNavDB
   （shelve，VLAD 索引/参考特征）。v1 行为：占用栅格/sdf/内参/prior npy 可读，
   重定位禁用并 WARN，`T_from_map_to_odom` 保持空 → nav_target_timer 早退
   （与 Python 自身守卫一致）。M2 内容。
6. teleop 依赖 pynput，镜像未装（离线），run_simulator.sh 已做跳过+WARN。
7. `--stack cpp` 与 `--map/--auto` 互斥（launch 暂无地图路径参数）——M2 后放开。
8. SDF 纹理路径已从 `file:///tinynav/...` 改为 `model://textures/...`（38 处），
   靠 `IGN_GAZEBO_RESOURCE_PATH` 解析；tinynav 上游若再改 SDF 需要同步。
9. **intra-process 与 /tf_static 冲突**：rclcpp 禁止 TRANSIENT_LOCAL 订阅在
   intra-process 下创建——planning 的 TransformListener 放在专用非 intra-process
   子节点（自带 spin 线程）上。mapping 的 TransformBroadcaster（/tf，volatile）不受影响。
10. rclpy 的 InputAligner（IMU/立体派发配对）未移植：IMU 直接回调处理，
    立体 worker 从同一把锁保护的 deque 消费——消费顺序契约保留（perception 组件头注释）。

## 待办计划

**已完成：**

1. ~~用户拍板的五步~~ ✅（planning 库 / 构建+冒烟 / 提交 / 对拍 / e2e）
2. ~~M2：map format v2（导出器 + C++ 读取器 + `--stack cpp --map`）~~ ✅
3. ~~perception v2：GTSAM 因子图对齐（GtsamRefine + 镜像内 gtsam 构建树）~~ ✅
4. ~~开销对比 v1/v2/python（栈口径，python 131.4% vs cpp 38.5/61.8%）~~ ✅
5. ~~脚下 ESDF 修复（min_wall_span_m=0.2 入 yaml）+ cpp 栈 rviz~~ ✅

**下一步待办（按优先级）：**

1. ~~reloc 匹配数 decisive 实验~~ ✅（img_shape 语义偏离，见"reloc 匹配数
   问题：已解决"；闭环验收已含在内：怠速+行驶 0 失败、reloc 位姿差 5mm）。
2. **TRT 引擎级对拍**：SuperPoint/LightGlue 输出逐值 vs python wrapper
   （探针已证双 wrapper 等价 + 逐位并发稳定，此条为固化回归防线）。
3. **场景套件 nightly**：l/u/z_corridor、factory_01 全链回归。
4. **reloc 后的闭环导航验收**：reloc 已通，下一步验证 POI 导航全链
   （/mapping/cmd_pois → global_plan → carrot 重投影 → 到达）。
5. 小项：`--stack cpp` 时 `pub_target.sh` 的 `-w 1` 匹配慢（手动发目标建议
   `-r 2 -t 4`）。~~5 个同名 /tinynav 节点让 `ros2 param` CLI 无法定位规划
   参数~~ ✅（2026-09-19 launch 去掉 `name='tinynav'` 覆盖，组件恢复真名；
   实验证实进程内通信按 topic+QoS 匹配、与节点名无关，改名后链路/CPU 无变化）。
6. **分机部署演练收尾**：~~a. B 侧数据面验证~~ ✅、~~b. 跨机行驶验收~~ ✅、
   ~~c. run_simulator.sh 定位~~ ✅（保留为单机开发全家桶，sim.launch 是
   分机/sim 独立面，见 sim/README）、~~e. 知识库回写~~ ✅（QA 6.9）。
   **d. 实机阶段**仍开放：USB NCM 直连（192.168.55.x）跑
   discovery_server:=对端IP，chrony 日志对时，链路带宽/丢包实测。

## 分节点日志（logsetup.py 的 C++ 移植，2026-09-19 落地）

对齐 python 侧 `<TINYNAV_DB_PATH>/logs/<YYYY-MM-DD>/<tag>.log` 布局（与
python 栈共树），全部语义以 `reference/tinynav/core/logsetup.py` 为 spec：

- **接入**：`logging_setup.cpp` 在 main() 安装一个 rcutils output-handler
  hook，按 logger 名把所有 `RCLCPP_*` 输出分流到各 tag 文件（perception /
  map / planning / imu_propagator / tinynav）——组件调用点零改动。文件
  DEBUG+ 全收，console 副本（stderr）INFO+ 同格式；控制台格式统一为
  python 的 RFC3339 本地时间 + ms（`2026-09-19T21:52:53.213 INFO map: ...`）。
- **轮转与保留**：跨天重开 `<day>/<tag>.log`；10MB 尺寸保险丝（
  `TINYNAV_LOG_MAX_MB`，glog 语义顺移 `.1/.2/.3.log`，`<tag>.log` 永远最新，
  0 关闭）；14 天清扫（启动一次 + 每 24h 守护线程，`RETENTION_DAYS` 常量）。
- **console 兜底**：fd 级 tee（dup2 + 泵线程，port of capture_console）把
  裸 printf/库输出也收进 `<day>/console.log`；stdout 设为非缓冲（对齐
  PYTHONUNBUFFERED=1 教训）。`TINYNAV_LOG_CONSOLE=0` 关闭兜底。
- **为何不用 spdlog**：`daily_file_sink` 是 final 且两种原生 sink 都表达
  不了"按天目录 + 尺寸保险丝"组合，最终 sink 就是 DayDirFileHandler 的
  1:1 移植（stdlib C++，零新依赖）；glog 无按天轮转、无保留清理（0.4.0
  实测），boost.log setup 成本高，均在调查后排除。
- **测试**：gtest 68/68（新增 7 条：tag 映射、格式契约、按天目录、追加、
  尺寸链顺移、清扫、缺根 no-op）；rig e2e：五文件落地、reloc/行驶/
  planning 心跳各行其是、console.log 镜像一致。
- **顺带修的坑**：`kill_sim.sh` 击杀名单漏了 `tinynav_node`——rig 重启后
  旧栈存活，双写者交错轮转把 `perception.1.log` 撑到 17.9MB（>10MB 上限），
  由此暴露并修复；排查时还踩到 `pkill -f tinynav_node` 自匹配杀掉调用
  shell（kill_sim 头注释里的旧坑，防了脚本自身没防调用者）。

## 对拍体系（已落地）

- 导出脚本（进 git）：`tools/export_planning_fixtures.py`、`tools/export_core_fixtures.py`；
  在容器内跑 reference 侧函数，产物写 `fixtures/`（gitignored）。
- 对拍测试（进 git）：`test/test_alignment_planning.cpp`（10 组场景：raycast 3、
  轨迹库 3+vocab、obstacle map 3、roll 4、footprint 3、ESDF 评分 5 输出、route
  fields 5、scipy EDT 距离+_indices tie-aware、DWA cost/argmin 3）与
  `test/test_alignment_core.cpp`（quat/rotvec/wrap、estimate_pose、
  pose_graph_solve vs 镜像 pybind .so、path_speed/path_climb）。
  fixtures 缺失时测试 SKIP，裸 checkout 不挂。
- 已知对拍边界：scipy EDT 与自实现 Felzenszwalb 的**等距最近点 tie-breaking**
  可能不同（edt.hpp 注释）——距离严格一致，remaining/heading 用 tie-aware 集合
  比较；estimate_pose 的 RANSAC 受 OpenCV 全局 RNG + 版本差异（pip cv2 4.11 vs
  系统 libopencv 4.5.4）影响，按真值恢复精度（1e-3）+ inlier 数量级对拍。
- 对拍揪出的移植 bug（已修）：
  1. build_route_fields 漏掉 `not any(inside) → has_route=False` 早退；
  2. build_obstacle_map 的 z_span 须按 **float32** 语义比较（Python z_idx 为
     float32 的伪影），double 比较在 0.2 边界会判反；
  3. 组件层 to_eigen_kpts/kpts_from_features 对引擎 [1,512,2] kpts 布局把
     size[2]=2（坐标维）当点数 N，越界读内存致 PnP 输入混入 -inf；
  4. match_indices 输出 dtype 是 **INT32→CV_32S**，at<double> 读全错。

## 端到端仿真验证（gzsim + --stack cpp，yard 世界）

- 引擎：镜像只带 ONNX，`cd /tinynav/tinynav/models && make dinov2 superpoint
  lightglue retinify` 一次性构建（TRT 10.13，RTX 5070 通过）。
- 已验证链路：gz 立体相机+IMU → C++ perception（立体深度 4Hz、SuperPoint/LightGlue
  匹配 ~126 对、PnP VIO → /slam/odometry_visual 4Hz、VIO failed=0）→ C++ planning
  （occupancy grid + ESDF → 目标 10m 时 sel vx=0.60 goal_err=0° front_clr=满、
  /planning/trajectory_path 4Hz）→ simulator_control（/cmd_vel 0.6 m/s）。
- `sim/robots/go2/spawn.sh` 增加 ros_gz_sim 缺失时的离线 fallback
  （ign sdf -p 转 SDF + ign service create）。

**镜像缺口（仅指 uniflexai/tinynav:latest；Python 栈同样受影响，非本仓库移植问题）：**

- 无 ros_gz_sim（create）——已用 spawn.sh fallback 绕过；
- 无 ros2_control / controller_manager —— joint_group_controller 无消费者，
  cmd_vel→关节执行断，机器人在 gz 里不能走（步态链）；
- 无 .plan 引擎（需 make all 一次性生成）；
- 空 __init__.py 的 `reference/tinynav/core/__init__.py` 与 AGENTS"不要加"冲突，
  实测无害（namespace 仍解析），是否删除待定。

**tinynav-runtime:x86_64 镜像的 rig 容器（实例名 `tinynav`）无上述缺口**：ros_gz_sim /
ros_gz_bridge / gz_ros2_control / controller_manager / 全套 .plan 引擎齐备，
go2 步态链完整可走；仓库挂载在 `/workspace/dm/tinynav-sim`（非 /ws），每次
切换容器需重跑 colcon build（CMake cache 绝对路径失效）。环境坑与调试工具
见下一节"rig 调试工具箱"，速查也写入仓库 AGENTS.md。

## rig 调试工具箱（2026-09-19 落地，全部实测）

reloc img_shape 排查会话的复盘产物——同类问题不再手搭脚手架：

1. **`sim/dog_state.sh`**：开工前一键确认机器狗状态（全局 AGENTS.md 纪律的
   工具化）。输出 gz 真值 + 自动换算的 yaw（度）；`--slam` 附 SLAM odom
   （odom 系与 gz world 系差固定安装偏置，实测 yaw -90°，两侧不要互比）；
   `--map-dir <v2图目录>` 对照建图轨迹包围盒（`pose_matrices.npy` x/y 平移
   min/max + margin 0.5m）给出 IN/OUT 判定与恢复提示。实测：狗开到
   (3.16, 1.84) 立判 OUT（y 越界），重启回原点判 IN。
2. **mapping 组件 dump 通道**：以 `TINYNAV_RELOC_DUMP_DIR=<dir>` 环境变量启动
   栈，每个被拒的 reloc 候选（LG <50）自动把该对完整 LG 输入写
   `<dir>/<live_ts>_<cand_ts>/`（map/live 六个 f32：kpts [1,512,2]、
   descps [1,512,256]、mask [1,512,1]（u8 转落盘 f32，引擎输入等价）+
   meta.json（匹配数/VLAD sim/img_shape/dtype）+ live.png；上限 50 对防
   失败风暴；env 不设零开销）。**回放**：`tools/probes/probe_lg <dump目录>`
   的 @848 计数应与 meta.json match_count 逐位相等（实测 36=36）。这是
   上节临时脚手架的常驻版。
3. **`tools/probes/`**（源码进 git）：probe_lg / probe_lg_race / probe_sp /
   probe_sp_race + probe_compare.py + capture_frames.py + `build.sh` 一键
   编译（依赖 colcon 树的 libtinynav_trt、CUDA/TRT/opencv4 头）。探针
   **数据**（dump、npy、png）仍在 gitignored `fixtures/probe/`。
4. **`tools/grab_topic.py`**：一发式抓话题存盘（任意类型存 rclpy 序列化
   .bin，sensor_msgs/Image 另解码 .npy，numpy-only 不依赖 cv2）。替代
   `ros2 topic echo --field data --raw`（对 keyframe_image 返回 0 字节）与
   每次重写的临时订阅脚本。

**`--stack cpp` 现在同样启动 rviz**（`/tinynav/docs/vis.rviz`，C++ 节点话题
面与 Python 完全同名，rviz 配置零改动）。

## 开销对比（C++ 栈 vs Python 原版实测）

条件：tinynav-runtime 容器，yard 世界 + go2，同一采样协议两栈各跑一遍
（15s idle → 发目标 → 80s 目标激活；top 每 2s 采样，稳态取均值）。

| 指标 | C++ 栈 v1（链式 PnP） | C++ 栈 v2（GTSAM，重测） | Python 原版（--stack full） |
|---|---|---|---|
| 栈 CPU（1 核=100%） | 38.5% 均值 / 87.6% 峰值 | **61.8% 均值 / 89.5% 峰值** | **131.4% 合计** = perception 105.9 + planning 25.5 |
| sim 桥接/控制（另列） | simulator_control 0.9 | simulator_control 0.9 | sim_gt_reloc 43.2 + simulator_control 0.9 |
| 内存 RSS | 2.37 GB（采样于运行 13 分钟时） | 1.13 GB（采样于运行 3 分钟时） | 1.62 GB 合计（1021+319+194+83 MB） |
| 线程数 | 48 | 51 | 104（52+28+12+12） |
| GPU | 19.4% util，584 MiB | 11.5% util（单次采样噪声大），584 MiB | 18.2% util，314 MiB（仅 perception 进程） |

- 栈口径说明：热路径只算两侧对应的节点进程——python 的 perception_node +
  planning_node（131.4%）；`sim_gt_reloc` 是 python full 栈无 map 模式下把
  gz 目标桥接进 SLAM 帧的 sim 专用脚本（C++ 栈目标直发 /control/target_pose，
  无此进程），`simulator_control` 两侧同跑，都另列不计入。C++ 单进程还多跑了
  mapping + imu_propagator 两个组件。
- v2 的 CPU 上浮 +23pp（相对 +60%）：窗口数据关联每帧 5 次 LightGlue +
  SuperPoint/LM 的开销，是 GTSAM 因子图对齐的代价；对比 python 为 ~1/2.1
  （v1 时 ~1/3.4）。
- GPU util 两轮采样都被 gz 渲染主导、路径不同，波动不可比；显存 584 MiB 与
  v1 完全一致（图求解全在 CPU）。

- v1 实测时（栈口径）：python 131.4% vs cpp 38.5% ≈ 3.4x，且 C++ 单进程还
  多跑了 mapping + imu_propagator。旧版表格曾把 sim_gt_reloc 43.2% 计入
  python 合计得出 175.5%/4.6x，口径错误，已废弃。
- GPU util 持平：两者都被 gz 渲染主导；C++ 显存高 ~270 MiB（TRT10 上下文/
  workspace 分配差异 + 三引擎常驻单进程）。
- Python 侧 RSS 合计低于 C++ 单进程，因为跨进程共享库按进程重复计入会被高估、
  且 C++ 侧 TRT 引擎 + 零拷贝缓冲常驻同进程。

## M2：map format v2（已落地；reloc 匹配数问题已解决，见后节）

已交付：

- 导出器 `tools/export_map_v2.py`：pickle poses + shelve（features/depths/
  vlad_descriptors/metadata）→ 纯 npy + meta.json，行序按时间戳升序。
- C++ 读取器 `mapping/map_v2.{hpp,cpp}`（`tinynav::mapping`）：f32/f64/i64/u1
  npy reader + `load_map_v2` + `keypoint_with_depth_to_3d` 移植。
- 组件接线：`load_map` 末尾加载 v2 → `relocalization_enabled_=true`；
  `keyframe_relocalization` 从 v1 骨架换成完整移植（VLAD top-3 候选 →
  LightGlue 匹配 ≥50 → 地图深度反投影 → rerank_by_pnp_inliers →
  FusionWindow + pose_graph_solve → T_from_map_to_odom）。
- 放开 `--stack cpp --map`：launch 加 `map_path` 参数，run_simulator.sh 加
  `--map/--map-dir`；`tools/build_map_live.py` 挂活栈直接采图（生产路径仍是
  rosbag + build_map_node.py）。
- 对拍：`tools/export_map_v2_fixtures.py` 合成 Python 格式图 → 导出 → gtest
  round-trip（poses/VLAD/features/depth 逐值）+ 反投影真值，60/60 绿。
- e2e 链路：build_map_live 采集（92 kf，x 0→6m）→ 导出 → `map v2 loaded`
  → 目标接受、自主行驶正常。

**[已解决，2026-09-19，见"reloc 匹配数问题：已解决"节] 当时的现象：reloc 候选的 LightGlue 匹配只有 1~25 对（门槛 50），从未成功**：

- 现象：每关键帧 3 个 VLAD 候选全部 "not enough matched features"；探针里
  map-vs-map（同图两帧）能到 208 对，map-vs-live 只有个位数到二十几。
- 已修一版：`match_keypoints` 原样沿用 python 写死的 `img_shape=[848,480]`
  （实机 D435 值），而 sim keyframe 是 544×480（gz 相机就是 544×480，fx=272
  可证）——改为传实际 keyframe 图像尺寸后仍只有 1~13 对，未解决。
- 待查方向：① map 特征出自 build_map_live 里 **python** SuperPointTRT、live
  特征出自 **C++** SuperPointTRT，两 wrapper 的 descps/mask 若有细微差异会直接
  压低跨库匹配（同一帧上两 wrapper 曾对拍到 126 对一致，但那是在 uniflexai
  容器/848 环境下）；② 544×480 小图上 SuperPoint 描述子质量与 50 对门槛的
  匹配性；③ descps 内存布局（[1,N,256] 行序）在 MapV2Features 与 live
  Features 两条填充路径上的等价性。
- 建议的隔离实验（未跑）：用 python map_node 直接挂本图（原生读 pickle+shelve）
  对同一话题流跑 reloc——若 python 也只有十几对，则是数据/分辨率问题；若 python
  能上百，则问题在 C++ 侧 wrapper/布局。

顺带修掉的两个休眠 bug（M2 首次激活相关路径时暴露）：

1. `tick_map_priors` 的 `(R·Pᵀ)ᵀ + t` 行广播求值在 **0 行**（不在爬坡区，
   常态）时是 Eigen UB，直接段错误——numpy 对 (0,3)+t 没问题，C++ 加了空守卫；
2. `match_keypoints` 图像尺寸（见上）。

**下一步（本阶段外，按之前方案推进）：**

- ~~GTSAM → perception v2 因子图~~ ✅（见下节）；
- ~~reloc 匹配数问题~~ ✅（img_shape 语义，见"已解决"节）；
- TRT 引擎级对拍（SuperPoint/LightGlue 输出逐值 vs Python wrapper）；
- 场景套件 nightly（l/u/z_corridor、factory_01）。

## reloc 根因探针记录（2026-09-19，rig 容器实测）

目的：判定 map-vs-live LightGlue 匹配仅 1~25 对（map-vs-map 208 对）是
C++ wrapper 问题还是数据/环境问题。探针源码现已迁至 `tools/probes/`
（git 跟踪，`build.sh` 一键编译；数据仍在 gitignored `fixtures/probe/`）：
`probe_sp.cpp`（C++ SuperPointTRT 落盘 npy）+ `probe_compare.py`
（Python wrapper 对比 + LightGlue 交叉匹配）。

**结论：双 wrapper 完全等价，跨 wrapper 假设被推翻；嫌疑转向地图数据本身
（内容/视角/填充路径）。**

| 测试图 | py vs cpp 关键点重叠(≤2px) | 描述子余弦 | LG py×cpp (848 形状) |
|---|---|---|---|
| 真实照片 960×720（丰富纹理） | 512/512 | 1.0000 | 502（= py×py） |
| gz 实拍 infra 帧 544×480（暗，std=11.5） | 512/512 | 1.0000 | 289（= py×py） |
| 对比度拉伸 / gamma 变体 | 512/512 | 1.0000 | 373 / 311 |

- N 两侧恒 512（动态引擎上限）、mask 全 true、kpts 布局同为 [1,N,2]；
  img_shape 848 vs 544 差异仅 502→505 量级，非根因。
- 过程中抓到一次假警报（probe_compare 读了上一张图残留的 npy，
  0/512 重叠），复跑即消失——探针对比必须每次重出 C++ 侧产物。
- mask dtype：Python 引擎输出 bool，C++ 输出 f32（值相同，全 1），
  喂 Python LightGlue 需 astype(bool)。

**剩余嫌疑（按可能性）：**
1. map v2 存储特征与 live 特征的内容/视角差异（208 的 map-vs-map 探针
   若用的是存储特征，则已证明存储特征内部健康，剩视角/内容一条线）；
2. mapping 组件 live 特征填充路径（to_eigen/kpts_from_features，
   即历史 bug #3 的同族代码）与 MapV2Features 两条路径的行序等价性；
3. build_map_live 采图时的图像源与 reloc 时的 keyframe_image 源不一致。

**下一步 decisive 实验**：把 VLAD top-1 候选的地图图像与触发失败的
live keyframe 图像并排存盘目检——图像明显同地不同视角 → 数据问题
（对策：建图覆盖视角 / 放宽 50 对门槛 / 建图也用 C++ wrapper 消除跨库
配对）；图像几乎相同仍 1 对 → 组件填充路径 bug，转引擎级对拍。

## reloc 匹配数问题：已解决（2026-09-19）

**根因：`match_keypoints` 的 `img_shape` 偏离了 python 语义。** python
（`map_node.py::match_keypoints`）在两个调用点都用写死的
`np.array([848,480])`（实机 D435 尺寸），无视实际图像尺寸；M2 时"修正"
为传 sim 实际尺寸 544×480，反而偏离 spec——LG 引擎内部的关键点归一化对
该值敏感，同一位姿对的匹配数在 544 vs 848 下差一截，恰好把大量边缘对压到
50 对门槛之下。复现的三组失败现场回放：

| 失败对（组件运行时 dump） | 组件@544 | 探针回放@544 | 探针回放@848 |
|---|---|---|---|
| f506023000000 × cand 80455000000 | 48 | 48 | **66**（过门槛） |
| f506221000000 × cand 79663000000 | 39 | 39 | 32 |
| f506221000000 × cand 80455000000 | 47 | 47 | **60**（过门槛） |

组件与探针@544 逐位一致 → 引擎/输入/代码全部无辜；848 才是 python 的
语义。**修复**：`match_keypoints` 恢复写死 {848,480}（含回环调用点），
见 `mapping_component.cpp` 内注释。

**验证链（全部逐位/数值对拍）：**
1. python map_node 旁证：同 cpp rig keyframe 流 + 原格式图，reloc
   **90/90 成功 0 失败**（且其 img_shape=848）。
2. 地图数据：shelve vs v2 npy 92 帧 kpts/descps/mask **逐位一致**。
3. 检索：cpp 与 python 对同一 live 帧的 VLAD top-3 一致，sim 差 ~9e-4
   （f32/f64 噪声级）。
4. 特征提取：cpp SP vs python SP 在失败帧上 descps/mask 逐位一致、
   kpts 差 6e-5px（rescale 舍入）；阈值两侧同为 5e-4。
5. 引擎执行：组件运行时 dump 的 LG 输入回放 = 组件所得（277/277、
   140/140、173/173、48/48、39/39、47/47）；LG/SP 两实例并发运行
   输出逐位稳定；match_threshold 两侧同为 0.1。
6. **闭环验收（修复后）**：yard，怠速 5 分钟（旧失败复现条件）+ 目标
   (6,0) 直线行驶：reloc 失败 **0**、VIO 失败 **0**，gz 真值
   (5.999, 0.042)，`/map/relocalization` 发布 (6.004, 0.005)——
   **reloc 纠正位姿与真值差 5mm/4cm**。

**顺带确认的三个既有事实（非 bug）：**
- 地图首 8 帧 + 末 14 帧 VLAD 描述子逐位相同（build_map_live 采集端对
  静止段的 VLAD 恰好逐位重复；python 同样容错，不阻塞 reloc）。
- 机器人**不在建图轨迹上/朝向反了**时 reloc 连败是正确行为（与 python
  一致，见 QA"定位丢失机理"）；本次排查中机器人曾偏到 y=1.95、yaw 104°，
  reloc 失败属预期。
- perception VIO 存在与 python 相同的结构性弱点：一帧 estimate_pose
  失败后 keyframe 窗口冻结，快速掉头（每 0.1° 一个 keyframe 的过渡帧）
  后可能死锁；`/slam/reset` 软复位可恢复（已验证）。是否加自动恢复
  （如连续 N 帧失败自动清窗）留作后续讨论项。

**遗留观察项**：旧 M2 会话日志（1~25 对）已被覆盖无法复核；按当前证据
其与 img_shape=544 + 边缘对组合吻合。若日后再现：mapping 组件的
`TINYNAV_RELOC_DUMP_DIR` dump 通道（见"rig 调试工具箱"）自动落盘失败现场，
`tools/probes/` 里的 probe_lg / probe_lg_race / probe_sp_race 直接回放，
另见 `sim/dog_state.sh` 先排除"狗不在建图轨迹上"的预期失败。

## perception v2：GTSAM 因子图对齐（已落地）

镜像自带完整 gtsam 4.3a1 构建树（/3rdparty/gtsam：源码 + libgtsam +
libgtsam_unstable + python 绑定），无需额外 docker 层。

- 接缝：`include/tinynav_cpp/gtsam/refine.hpp`（tinynav::gtsam::Refine；
  RefineInput 把窗口时间戳/IMU 样本/UF tracks/速度先验索引交给实现，gtsam
  类型不出现在组件里）；`src/gtsam/gtsam_refine.cpp` 完整移植 [ISAM
  Processing] 块：bias 先验(1e-2)、首帧 pose 先验(1e-1)、相邻帧
  CombinedImuFactor、失败 pair-PnP 的速度先验(0.25)、每 track 一个
  SmartStereoProjectionPoseFactor(Isotropic 1.0 + 默认 SmartProjectionParams +
  Cal3_S2Stereo)、LM 3 迭代，结果写回 pose/velocity。
- CMake：`TINYNAV_GTSAM_ROOT=/3rdparty/gtsam` 存在则编 tinynav_gtsam 静态库并
  链入 perception_component；**RUNPATH 必须设在 perception_component.so 上**
  （静态库自己的 INSTALL_RPATH 不会传给消费者对 libgtsam.so.4 的查找，否则
  launch 起不来：error while loading shared libraries）。无 gtsam 的机器自动
  降级 v1 链式 PnP。
- 数据关联留在组件侧（SuperPoint 逐关键帧缓存进 Keyframe、LightGlue 逐对、
  estimate_pose 内点过滤 >20、core::UnionFind 组 track、视差 ≥0.1 检查，
  _M=1000 stride），实现侧只建图求解。
- 对拍：`tools/export_gtsam_fixtures.py` 用 python 绑定解同一场景（4 关键帧 +
  100Hz IMU + 3 tracks + 速度先验），C++ refine 复现：**初始/最终误差与 python
  完全一致（1950.900853→330.124488），位姿 1e-8**；61/61 测试绿。
- rig 实测：GTSAM refine enabled，469 因子/15 变量，误差 0.0003（python 同
  rig 0.0023 同量级），VIO failed=0，/slam/odometry_visual 4-6Hz。

移植时踩掉的三个 rig 专属坑（对拍测不出来，实跑暴露）：

1. `core::ImuSample` 字段顺序是 {stamp, **gyro, accel**}，组装时 accel/gyro
   装反 → 预积分全错、CombinedImuFactor 误差 inf；
2. python 的 drain 有 `timestamp <= latest → continue` 守卫，乱序 IMU 戳会让
   dt<0、协方差变负（inf）——批量预积分补了同样的守卫；
3. python 逐到达积分不依赖 imu deque 保留历史，批量方案需要完整窗口跨度：
   deque 上限 1000（10s）会驱逐旧样本 → 老窗口对零样本 → inf。上限提到
   4000，refine 后按窗口起点修剪。

已知分歧（有意为之，见 RefineInput::imu 注释）：python 每帧 drain 会把 peek
到的那条样本重复积分（每帧约一个样本区间 <10ms 的双重计入），批量方案每条
样本恰好积分一次；两侧 batch 约定一致，对拍不受影响。

## 分机部署（x86 = sim+perception ↔ Orin = bridge+三件，2026-09-20 实施中）

背景与拍板：perception 将跑在相机（实机 = Looper Insightfull 盒子，仿真 =
x86 上的 gzsim）内部，Orin 只留 imu/mapping/planning 三件。跨机链路复用
fleet 的 looper_bridge 模式——**链路上每个话题只允许一个订阅者**（DDS 单播
按订阅者复制，桥是唯一订阅者，录包/监控/调试 tap 本地输出零链路成本），
**改名重发**防同名双匹配（桥若原名重发，Orin 订阅者会同时匹配远端+本地桥，
双份且混源）。

**与原版 looper_bridge 的关键差异（读了 pilot 仓库
`tinynav/tool/looper_bridge_node.py` 当 spec 后拍板）**：原版是**处理型桥**
——订阅相机盒的逐帧三话题（depth/vio_image/infra1）做 TimeSynchronizer(20)，
**keyframe 判选（0.03m/1°/3s）就在桥里**，还做深度 16U→32F 米制转换、
vio_100hz→/slam/odometry 的 PoseStamped→Odometry 转换、infra1 camera_info
别名重发为 infra2。因为 fleet 的相机盒自己跑 VIO。我们拓扑里"相机"= 自家
perception 组件，产物已是最终形态 → 桥退化为**纯中继**（改名+QoS，盖章
透传），发布契约与原版对齐（同名正名、reliable sensor_qos）。imu /
infra2 camera_info / /clock 三条小话题**不走桥**：单消费者、~60KB/s，
直连；桥若转发反而制造双份（不变式：任一跨接话题 Orin 侧订阅者 ≤1）。

**交付物（全部已实现，构建绿，gtest 68/68）：**

- `main.cpp` 组件子集：`TINYNAV_COMPONENTS=imu,perception,mapping,planning`
  逗号表（缺省=全栈），未知名字报错退出。x86 跑 `perception`，Orin 跑
  `imu,mapping,planning`；单进程 intra-process 架构不变。
- `src/bridge_node.cpp`（tinynav_bridge 可执行）：YAML 驱动的中继表
  （订阅名→发布名→QoS），盖章透传，独立进程（链路端点，栈崩链路不死，
  fleet 教训）。转发表 `config/link_relay.yaml`：5 条 /camera/camera/slam/*
  → /slam/*（odometry_visual、keyframe_odom/image/depth reliable，
  depth reliable——planning 的 latest_depth_only 订阅是 RELIABLE，QoS 必须
  配平，实测 best_effort 会被 rmw 拒配对）。
- 三个 launch：
  - `sim/launch/sim.launch.py`：sim 层独立启动（gz server/gui、spawn、
    gz bridge、camera_info、simulator_control、teleop 可选），spawn 位姿表
    与 bridge 话题表从 run_simulator.sh 移植；**含 discovery server 进程**
    （`python3 /opt/ros/humble/tools/fastdds/fastdds.py discovery -i 0`，
    注意 bin/fastdds 是无 shebang 包装脚本，launch exec 会 Exec format
    error，必须直调 python 入口）。
  - `tinynav_cpp/launch/perception.launch.py`：perception 子集 +
    /slam/*→/camera/camera/slam/* 出站重映射（发送侧进相机盒命名空间）。
  - `tinynav_cpp/launch/orin_stack.launch.py`：bridge + 三件一进程 +
    use_sim_time:=true（时间一致性走 /clock 过链路 + sim 时钟，不靠机器
    对时；chrony 只为跨机日志对时）。
- `tools/probes/probe_first_msg.py`（数据面订阅探针，类型参数
  Image/Odometry/Imu/CameraInfo/Clock）与 `probe_pub_once.py`（数据面发布
  探针，RELIABLE，替代 ros2 topic pub）——DS 模式下唯一可信的验证/发令
  手段。

**DDS 拓扑（两容器演练调通，实机同构）：** `FASTDDS_BUILTIN_TRANSPORTS=
UDPv4`（禁 SHM——跨容器 /dev/shm 隔离，SHM locator 被通告但不可达，图
fflap；实机跨主机本无 SHM，UDP-only 让演练=部署）+ `ROS_DISCOVERY_SERVER
=<sim站点IP>:11811`（单播发现——点对点链路/wifi 组播不可靠）。演练实测
demo talker/listener 跨容器互通。

**两容器演练结果（2026-09-20，全链通过）：**

- ✅ A 侧：sim（yard+go2）、perception 重映射产物 /camera/camera/slam/*
  数据面探针实测到达（VIO ISAM 正常求解）。
- ✅ B 侧四探针（probe_first_msg，真实订阅）：/camera/camera/imu（直连）、
  /slam/odometry_visual + /slam/keyframe_image + /slam/keyframe_depth
  （桥转发，544x480 mono8 / 32FC1 无损）、/slam/odometry（imu_propagator
  融合输出）全部到达。
- ✅ 跨机行驶：B 侧 rclpy 探针发 /control/target_pose (6,0)，狗开到
  (6.09, -0.07) IN——同时验证了反向链路（/planning/trajectory_path 回
  A 的 simulator_control）。
- ✅ reloc 分机验证：/map/relocalization 在 B 发布 (6.082, -0.040)，与 gz
  真值 (6.087, -0.074) 差 ~3cm（单机验收同级精度）。
- 演练环境：容器 A（tinynav，挂 /home/dm/workspace→/workspace）跑 sim +
  perception + discovery server；容器 B（tinynav-orin，同样整 workspace
  挂载）跑 bridge+三件，日志根 output/orin_db/logs/。

**排查过程抓到的重要坑（后续会用上）：**

1. **Humble ros2cli 在 Discovery Server 客户端模式下全盲**：`ros2 node
   list`、`topic list`、`topic hz`、`topic info` 对实际在通信的参与者
   一律报空/未发布（demo talker/listener 同环境实测数据互通，CLI 同时
   瞎）。分机验证必须用真实订阅（probe_first_msg / demo listener），
   绝不能拿 CLI 输出当"链路不通"的证据——本次排查多轮假阴性皆源于此。
2. fastdds CLI 链：`bin/fastdds`（无 shebang shell 包装）→ `tools/
   fastdds/fastdds.py` → 子进程 `bin/fast-discovery-server`。**杀 wrapper
   不杀子进程**：孤儿 server 继续占 11811 端口，后续 launch 的 server
   "wasn't able to allocate the specified listening port" 全灭。清场必须
   `pkill -f fast-discovery-server`（kill_sim.sh 尚未收录，待办）。
3. 容器 PID 1 不收割僵尸：pgrep/pkill 会命中 defunct 进程造成"杀不死/
   还活着"假象；且 `pkill -f <串>` 会匹配调用者自身命令行（docker exec
   bash -c 的整串），多次把排查 shell 自己杀掉（exit 137/143）。清场用
   kill_sim.sh（独占一条调用），存活判断看数据面/日志而非进程表。
4. FastDDS 2.6 XML profile：xmlns 命名空间会让解析器报 "userTransports
   without content"（profile 整个被丢且只打一行解析错误）；且 Humble 有
   `FASTDDS_BUILTIN_TRANSPORTS=UDPv4` 环境变量，根本不需要 XML——本次
   最终方案纯 env，零配置文件。
5. Humble launch 的 `ExecuteProcess`：cmd 是强制关键字参数（`cmd=[...]`），
   位置传参直接 TypeError。
6. ros_gz_bridge 每条话题默认建**双向**桥（GZ→ROS + ROS→GZ 都创建）。
7. **引擎目录软链坑**：rig 镜像 `/tinynav -> /workspace/dm/tinynav-pilot/
   tinynav`（软链），容器若只挂仓库子目录则软链悬空 → TRT 引擎**静默
   降级**（"dinov2 engine unavailable: loop closure disabled"，进程不崩、
   reloc/闭环无声禁用，只剩 10s 节流的 WARNING）。演练容器必须挂
   /home/dm/workspace→/workspace 全路径（或对齐软链指向）。

**实机部署还差的事（对应待办 6d）：** Orin 侧 USB NCM gadget 直连
（192.168.55.1/100，即插即用，L4T 默认带）；两侧
`ROS_DISCOVERY_SERVER=<x86 链路 IP>:11811`（discovery server 固定跑 x86）；
chrony 对时（只影响日志对时）；链路实测丢包后若 keyframe trio 掉帧，
考虑调 FastDDS 分片/buffer（XML 方案备好再上）。

## 验证命令备忘

```bash
# 全量构建（容器内）
docker run --rm --gpus all --network host -v "$PWD":/ws -w /ws \
  uniflexai/tinynav:latest bash -c \
  'source /opt/ros/humble/setup.bash && colcon build --packages-select tinynav_cpp'

# 跑全栈（单进程 4 组件）
docker run --rm --gpus all --network host -v "$PWD":/ws -w /ws \
  uniflexai/tinynav:latest bash -c \
  'source /opt/ros/humble/setup.bash && source install/setup.bash && \
   ros2 launch tinynav_cpp tinynav.launch.py'

# 仿真（python 栈 / cpp 栈）
docker run --rm --gpus all --network host -v "$PWD":/ws -w /ws \
  uniflexai/tinynav:latest bash sim/run_simulator.sh --robot go2 --world factory
docker run ... bash sim/run_simulator.sh --stack cpp --robot go2 --world empty
```

## 2026-09-23 关键帧降密 + map_v2 lazy + LiveCapture 落盘 + 深度 Z16

背景：艺尚狗实测 8 图 110G（单图 22G = 9587 帧 × f32 深度 13G + SP 特征 4.9G），
关键帧阈值 0.03m≈逐帧存；map_v2 eager 物化在艺尚级真图上 ~18G RSS 必 OOM；
组件自身帧内存 map 随任务时长无界增长（nav_temp 的"内存替代"偏离了 python 语义）。

- **关键帧阈值** `perception_component` 0.1m/0.1° → **0.3m/10°**（3s 超时保留）。
  yard 实测行驶 ~1Hz（旧 ~6Hz）。狗上 looper_bridge 的 `--keyframe-translation`
  默认 0.03 也要同步改（现场侧待办）。
- **map_v2 lazy**：`mapping/mapped_npy.hpp`（只读 mmap npy，MADV_RANDOM）；
  MapV2 eager 只留 timestamps/poses/VLAD 索引，depth/features 走
  `get_depth/get_features` 零拷贝行视图。加载艺尚级图从 ~18G RSS → 近基线。
- **LiveCapture**（`mapping/live_capture.{hpp,cpp}`）：自身帧落盘，python
  nav_temp_db 语义对齐——开即清空（scratch）、embeddings 留 RAM（find_loop
  每帧扫描）、depth/features 列式追加按候选读。布局 = v2 文件集增量写
  （128B 固定 npy 头 open 预写/close 原地补 shape），**close 后目录 + poses +
  VLAD 就是合法 v2 子集**（测试经 load_map_v2 回读锁死该性质）。接线：组件
  `features_`/`depth_of_` 内存 map 删除，append 失败的帧不参与回环（不索
  embedding）；`live_capture_dir` 参数默认 `nav_temp_v2`。
- **深度 Z16**：v2 `depth_images.npy` 改 **u16 毫米**（导出器 + LiveCapture 一致；
  reader 兼容旧 f4），MapV2/LiveCapture get_depth 按候选帧转回 f32 米。
  狗上 13G depths.db 同源改造后 ~6.5G。**夹具 expected 仍为 f32 米真值**，
  测试比较放行 ±0.5mm 量化。
- **测试 72/72**：新增 live_capture 两个（RSS 平坦 + u16 往返 ±0.5mm + close
  目录 load_map_v2 读通；shape 突变拒帧不停会话）；压力图改 4000 帧 u16 2.6G。
- **e2e（yard 单机 cpp --map u16 图）**：reloc 锁定、两趟往返 140m 到点精确、
  append 0 失败、RSS 737→854→797MB（自身帧在盘，内存零增长）、live 目录
  130M/55 帧。坑：close 后 1 维 npy shape 曾写成 `(N,,)`（C++ 解析宽容、
  numpy 拒读，破坏 v2 互操作）已修——**改格式后必须用 numpy 回读验证**。
- 环境坑：镜像 ENV 烤死的 `CYCLONEDDS_URI=/tinynav/scripts/cyclone_dds_
  localhost.xml` 在 rig 容器不存在 → 所有 Cyclone 节点建域即死；run_simulator.sh
  已统一覆盖为空（分机 launch 自带 URI 不受影响）。

### 2026-09-23 补：VLAD 检索索引 f64→f32（8G Orin Nano 预算）

狗实为 Orin Nano 8G，f64 索引同密度 1.9G 不可接受。`MapV2::vlad_descriptors`
改 **f32 行主序**（`VladIndex`，npy f32 直接 memcpy；f8 输入降精度兼容），
查询向量一次性 cast f32 后点积（f32 带宽减半）。`vlad_centres` 保持 f64
（32x768=200KB 无关紧要，compute_vlad 库零改动）。python 本来就是 f32——
C++ 的 f64 反而是保真度偏离，此改同时修正。加载日志新增索引 MB。
测试 72/72；e2e：`VLAD 92x24576 f32 (9MB)`、到点 4.48/4.5、reloc 锁定零失败。
8G 预算（降密后 959 帧）：索引 94MB + 位姿/杂项 ~30MB + TRT 引擎与 VIO 若干百 MB
——eager 索引无压力；同密度 9587 帧则索引 943MB，仍以降密为先。
