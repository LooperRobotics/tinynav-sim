# HIL 计划：gsplat 仿真 × navcore（looper 契约仿真器路线）

> **执行状态（全部落地，比预估 3.5-4 天快得多）**：
> - §2.0 契约采样 ✓ → `docs/looper-contract.md` + oracle 探针 `docs/oracle/` +
>   012 机上 401s 参考 bag（`~/hil_step0/bags/looper_contract_60s`，8.7G）
> - §2.3 ✓ ExactTime 确认（三元组必须严格同 stamp）
> - §2.4 ✓ `docker/arm/deploy/{docker-compose.hil.yml,hil_fastdds.xml,hil_on.sh,hil_off.sh}`
>   （deploy kit + 012 机上各一份），跨机冒烟双向通过
> - §2.1 ✓ ring v3（depth u16mm + gt_cam 槽）+ sensor_server `--cam-w/--cam-h/--cam-fy/--cams`
> - §2.2 ✓ `gsplat/ros/looper_emu.py`（节点名 insight_full，三元组同频同 stamp 直读 ring）
> - §2.5 阶梯 1-3+cmd_vel 直行 ✓（sensor:True、/slam 流、狗按 cmd_vel 走 0.92m）；
>   建图→带图导航留给前端实操；hil_off 回归 ✓
> - 启动：x86 容器内 `bash gsplat/run_hil.sh`（停：kill_hil.sh）；navcore 侧 hil_on/hil_off
>
> **实现期四个大发现（后续维护必读）**：
> 1. **~~fastdds_bind.py 白名单从未生效~~（订正：误判）**：当时只 grep
>    了新名 `FASTDDS_DEFAULT_PROFILES_FILE`，漏了 compose 直配的老名
>    `FASTRTPS_DEFAULT_PROFILES_FILE`（FastDDS 向后兼容仍生效）——**各 rig 的
>    白名单一直是加载的**（晋南机 9 天 1.96TB/0 丢包 + 生产 bag 满帧佐证）。
>    hil 层自带 XML 的真实理由只剩一个：hil 白名单（l4tbr0 单绑）与 rig 白名单
>    （enx）指向不同文件，需要 env 显式指过去。（老名→新名仍是值得做的现代化）
> 2. **内核级泄漏（白名单挡不住）**：looper 的 writer 学过 pilot 读取器的
>    `192.168.55.1:<RTPS 确定性端口>` 后，经默认网关持续单播推数——包从 enx 进、
>    目的 55.1 是本机地址，弱主机模型下照收。症状：vio_status 双流交替、
>    odometry_visual 频率翻倍（三元组被双源穿插）。闸门在内核：
>    `iptables -I INPUT -s 169.254.10.0/24 -d 192.168.55.1/32 -j DROP`
>    （hil_on/hil_off 已内置增删）。
> 3. **设计变更：emu 直读 ring，不起 gs_ros_bridge**——两者同发
>    /camera/camera/infra1/*，同跑必双源打架；且省一次 DDS 序列化往返。
>    `run_hil.sh` 只起 gssim+emu 两个窗口。
> 4. **内参偏差（自洽即可）**：batch_render 强制 cx=W/2、cy=H/2；emu 的
>    camera_info 按居中模型声明（fx=302.779, cx=272, cy=320），与 oracle 的
>    cx=267.59 差 4.4px——图像与声明自洽，navcore 行为不受影响。
>    相机挂载外参由 emu 前 10 帧自学习（静止期 gt×cam 配对），非手配。

> 目标：x86 跑 gsplat 3DGS 仿真"冒充" looper 相机盒，USB 连接的 navcore（orinnano-012，
> pilot 栈 rig 形态）**零改动**地完成硬件在环——navcore 上跑的是生产同款 pilot 行为树 +
> mapping/planning，VIO 由 GT 位姿顶替（与真机架构一致：VIO 本来就在相机盒里）。
> 决策记录：路线 B（navcore 不动）优于 A（pilot 切 realsense 模式）——A 需遮
> run_realsense_sensor.sh 硬件驱动脚本；本计划 DDS 改动走 compose 叠加层，回真机 = 一条标准命令。

## 0. 架构

```
[x86 工作站, tinynav-sim 容器(host net)]
  gsplat sensor face（已存在：run_gsplat.sh --stack sensor）
    └─ infra1/infra2/color + camera_info + /sim/gt_pose + /clock
  looper_emu（新增桥，节点名必须 = insight_full）
    ├─ 直通：infra1/image_rect_raw、infra1,2/camera_info（补 frame_id/QoS 对齐）
    ├─ depth：gsplat 渲染深度 → /camera/camera/depth/image_rect_raw
    ├─ vio_100hz：/sim/gt_pose → 100Hz 重发（PoseStamped, T_world_camera）
    ├─ vio_image：GT 位姿按 keyframe 判选（0.03m/1°/3s，抄 fleet 阈值）→ PoseStamped
    │   ※ 三元组 (depth, vio_image, infra1) 必须同 stamp 发布（见 2.3 同步策略）
    └─ /tf_static：一帧 TFMessage（内容从活体 looper 抄，见 step 0）
         │ FastDDS（组播先试 / DS 备胎）
         ▼ USB gadget 192.168.55.0/24
[navcore orinnano-012, pilot rig 栈（仅 DDS 叠加层临时改绑 l4tbr0）]
  core-runtime: looper_bridge_node.py（一字不改）
    └─ 契约消费 → /slam/* → mapping/planning/行为树
  仿真狗控制：planning → /cmd_vel → x86 RL 步行策略
```

## 1. 关键事实（已核实，实现时不要重查）

- **looper 契约**（looper_bridge_node.py 订阅面，navcore 端唯一消费者）：
  `/camera/camera/infra1/image_rect_raw`(Image) + `/camera/camera/depth/image_rect_raw`(Image)
  + `/camera/camera/vio_image`(**PoseStamped**，非图像！) 三元组 message_filters 同步；
  `/camera/camera/vio_100hz`(PoseStamped, depth=50)；`infra1/infra2/camera_info`(CameraInfo)；
  `/tf_static`。输出 `/slam/{odometry,odometry_visual,depth,disparity_vis,camera_info,
  keyframe_odom,keyframe_image,keyframe_depth}`。sensor_qos = RELIABLE, depth=50。
- **pilot 的传感器就绪判定**（sensor.py）：`sensor_source_ready()` 在 looper 模式下等
  **ROS 图里出现节点名 `insight_full`** → **looper_emu 的节点名必须叫 /insight_full**，
  否则 pilot 卡 baseline 等待（超时后虽会带起，但 WatchSensorProc 会常红）。
- **imu 话题无人消费**（logs.md 实测 rig 上 imu 0 读者）→ emu 可不发 imu，省带宽。
- **时间戳**：GT 路线下 navcore 不做状态估计时间积分（位姿直接给），stamp 只影响
  三元组同步与 keyframe 节奏 → emu 内部统一用** x86 墙钟**盖戳、保持单调即可，
  不发 /clock、navcore 不设 use_sim_time（生产保真）。
- DDS：core-runtime 白名单钉 `169.254.10.2+127.0.0.1`（真机 looper 链路 L2 邻接才通），
  x86 够不到 → 必须临时改绑 l4tbr0；回环 127.0.0.1 恒在白名单，容器间互通不受影响。

## 2. 实现分解

### 2.0 契约采样（step 0，活体 oracle，~0.5d）
桌上这台 navcore 正挂着**真 looper**（deep-flow on）——趁拆前把契约钉死：
- `ros2 topic info -v` 全量：每话题 type/QoS/rate；`ros2 topic echo` 抓 frame_id 集合
- `/tf_static` 完整内容、vio_image/vio_100hz 的 frame_id 与数值语义（是否 T_world_camera）
- 录 60s 参考 bag（含全部契约话题）供 emu 输出比对
- 产出 `docs/looper-contract.md`（表：话题/类型/QoS/频率/frame_id/备注）

### 2.1 gsplat depth 通道（~0.5-1d）
sensor_server 渲染循环（15Hz、渲染预算 66.7ms）加深度合成输出进环形缓冲；
ros bridge 增发 `/camera/camera/depth/image_rect_raw`（16bit? 对齐 oracle 编码）。
注意渲染预算——depth 复用同一 splat raster，增量小。

### 2.2 looper_emu 桥（~1d，tinynav-sim 新组件 `gsplat/ros/looper_emu.py`）
- 节点名 `/insight_full`（硬要求，见 §1）
- 订阅 face 的 infra1 + gt_pose +（新）depth；发布契约全量
- ~~keyframe 判选~~ **（step0 修订：不需要**——真 looper 逐帧发 vio_image 20Hz，
  0.03m/1°/3s 判选本来就在 looper_bridge.should_add_keyframe 里，桥不动即继承。
  emu 只做三元组同频同 stamp 直发（20Hz），命中率 100%）
- vio_100hz：10ms 定时器重发最新 GT 位姿（起步不做插值，够用）
- QoS/分辨率按 docs/looper-contract.md 逐项抄：544×640、mono8/mono16、
  fx=302.779、tf_static 五条照抄、vio_status TRANSIENT_LOCAL 1Hz

### 2.3 同步策略确认（读代码，~0.2d）
确认 looper_bridge 三元组用的是 ExactTime 还是 ApproximateTime（源码 lines 52-56 的
message_filters.Sync policy）——若 ExactTime，emu 必须严格同 stamp；若 Approximate，
容差多少。**实现顺序上放在 2.2 之前读掉**，避免返工。

### 2.4 DDS 叠加层 + hil_on/hil_off（~0.5d，产出进 docker/arm/deploy/）
navcore 机上 pilot 目录放 `docker-compose.hil.yml`（仅 env 差量）：
```yaml
services:
  core_runtime:
    environment:
      FASTDDS_BIND_REQUIRED: l4tbr0
      # 组播不通时解开（DS 备胎）：
      # FASTDDS_BUILTIN_TRANSPORTS: UDPv4
      # ROS_DISCOVERY_SERVER: "192.168.55.100:11811"
```
- `hil_on.sh`：COMPOSE_FILE 三层显式 up -d core-runtime + POST deep-flow off
  （`http://169.254.10.1/api/deep-flow {"enabled":false}`）+ DNS can0 自愈检查
  （`resolvectl revert can0` 那条，looper 操作后复发坑）
- `hil_off.sh`：**裸** `docker compose up -d core_runtime`（默认只合并 yml+override，
  hil 层自然出局——真机恢复零记忆负担）+ deep-flow on + 验证 rig 传感器流恢复
- 不变量：真机真值永远只在 override.yml；hil 层永远显式追加

### 2.5 联调与验证阶梯（~1d）
1. **x86 单机自测**：`run_gsplat.sh --stack sensor` + looper_emu，本地订阅者验证
   契约话题齐、三元组同步命中率 100%、节点名 = insight_full
2. **跨机冒烟**：hil_on 后，navcore 宿主机 `ros2 topic list` 应看到 x86 的
   /camera/camera/*（组播路线 CLI 可用）；不通 → 解开 DS 两行 + x86 起
   `python3 /opt/ros/humble/tools/fastdds/fastdds.py discovery -i 0`
   （注意：DS 模式下 ros2cli 全盲，验证只能用真节点探针——probe_first_msg）
3. **pilot 消费**：`docker logs core-runtime` 看 looper_bridge 三元组首中；
   `:8100/status` sensor:True、`/slam/*` 有流量
4. **端到端**：teleop/cmd_vel 直行 → 建图（/mapping/start + stop 出图）→ 带图导航
   （/control/target_pose 发目标，gs_state.sh --slam 看 IN/OUT）
5. **回归**：hil_off → 真 looper 流恢复（violence 检查 vio_image hz + sensor:True）

## 3. 风险与备胎

| 风险 | 缓解 |
|---|---|
| USB gadget 链路组播不可靠（当年相机盒链路的教训） | DS 备胎已内建（两行 env + x86 server），切换零成本 |
| vio_image 语义理解偏差（PoseStamped 的 frame/方向） | §2.0 活体采样直接抄数值语义；emu 输出与参考 bag 对拍 |
| /tf_static 外参错误 → 桥内 TF 变换错 | 从活体 looper 逐字节抄，不自造 |
| splat 深度噪声高于真相机 | planning 消费的是桥内 keyframe_depth；记录噪声水平，必要时对 depth 做中值滤波 |
| WatchSensorProc 判定依赖 bridge_out_hz | emu 喂对契约则 /slam/* 有流量即绿；若仍红查 topic_rates 探针话题名 |
| DNS can0 复发（looper API 操作后） | hil_on/off 内置自愈检查 |
| x86 容器混杂（当前有个 idle 的 `tinynav` 容器占名） | 联调前清场：gsplat 用 tinynav-sim compose 容器 |

## 4. 工作量与顺序

契约采样(0.5) → 同步策略确认(0.2) → depth 通道(0.5-1) → looper_emu(1) →
DDS 层+脚本(0.5) → 联调(1) ≈ **3.5-4 天**。其中 2.4（DDS 层）可最先做——
不依赖仿真件，先用 talker/listener 验证跨机链路。

## 5. 明确不做（scope 外）

- 不动 navcore 任何代码/镜像/仓库文件（唯二落点：机上 hil.yml + deploy kit 两个脚本）
- 不仿 imu 话题（无消费者）；不做 use_sim_time / /clock 传播（GT 路线不需要）
- 不做 A 路线（realsense 模式感知上板）——将来要验证板端感知时另立计划

## 6. 已知问题：navcore 在线录制丢帧 —— **已解决（CycloneDDS）**

**现象**：录制时 bag 里图像只剩 3/660 帧（小话题全到），bridge 输出出现 0.9s 突发空洞。

**根因链（全部实测钉死）**：
1. **管道丢包**：x86 TX 171.5GB vs 012 RX 163.0GB（8.5GB 丢在 navcore 当 USB 设备的
   NCM gadget 接收路径，netdev 零计数）。TCP 78MB/s 无损——只丢突发 UDP 洪水。
   生产 looper 链路（navcore 当 USB 主机）同流量零丢 → 该问题为 HIL 特有暴露。
2. **FastDDS 默认 65KB 数据报**（抓包实证，真 looper 同款）：44 个 IP 分片/报，
   5% 分片丢失下整报存活率 0.95^44≈10% → RELIABLE 重传风暴。
3. recorder 为 RELIABLE/VOLATILE（可恢复），但恢复追不上 writer 历史覆盖。

**实验矩阵结果**（一次一个变量，35s 帧计数）：
| 配置 | 图像帧到达率 |
|---|---|
| 基线（FastDDS 默认） | 16% |
| NCM MTU 15436 两端 | ✗ 主机侧驱动被 gadget 描述符钉死 1500，判死 |
| FastDDS maxMessageSize 1400（RTPS 级分片） | 40% |
| + writer history depth 20→200 | 39%（无变化，瓶颈在管道负载相关丢包） |
| **emu 换 CycloneDDS（FragmentSize 1344/MTU 适配）** | **100%** |

**胜利配置**：emu 走 `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` +
`gsplat/configs/hil_cyclone_x86.xml`（接口只列 55.100，**绝不列 127.0.0.1**——
Cyclone 会选 lo 当组播口并禁组播，参与者全网不可见；也不吃镜像烘焙的
localhost-only CYCLONEDDS_URI）。与 navcore 侧 FastDDS 跨厂商 RTPS 互通。

**E2E 验收（全绿）**：pilot `/bag/start`→遛狗→`/bag/stop`→
`/map/build` = **rc=0，483 poses 的真图**。bag 内 infra1/color 100%、depth 91%。
emu 补发了 `color/image_rect_raw/compressed` + `color/camera_info`（非契约但
build_map_node 四路同步必需：keyframe_image/odom/depth + color raw，由
ImageTransportsNode 解压——真 rig recorder 录 15 话题，此前 bag 缺 color 导致空图）。

**排障中沉淀的三个坑（复发必查）**：
1. Cyclone 接口表含 127.0.0.1 → "selected interface lo...disabling multicast" →
   全网不可见；且 pilot 的 sensor:True / TRACKING_GOOD 是缓存假象——**判活必须
   实测 /slam/odometry_visual 流量**（探针见下）。
2. core-runtime 里的 looper_bridge 可能因传感器重试陷入僵死（CPU 0%、无输出、
   无日志——stale SHM + rclpy C 层重试卡 GIL，compose 注释记载的老病）；
   `docker restart core-runtime` 是可靠复位。
3. `error: build` 状态是故意粘性的（前端要显示）——除 pilot 重启外不消失；
   /bag/start 的路由守卫会因此拒绝，别跟状态机较劲，直接重启。
   探针：`python3 /tmp/probe_any.py /slam/odometry_visual Odometry`（容器内，
   带 hil 白名单 env；注意 ros2 topic hz 的 CLI 探针在此环境输出会静默丢失）。
