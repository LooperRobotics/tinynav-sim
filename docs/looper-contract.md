# Looper 相机盒 DDS 契约（活体采样）

> 采样环境：orinnano-012（pilot rig 形态）挂真 looper，deep-flow **on**
> （perception_engine=depth），ROS_DOMAIN_ID=0。采样手段：宿主机 ROS2 Humble
> + FastDDS 白名单 XML（`~/fastdds_looper.xml`，169.254.10.2+127.0.0.1）。
> 参考 bag：`~/hil_step0/bags/looper_contract_60s`（012 机上，实际 401s/8.7G——
> `-d` 是切片时长不是总时长，SIGINT 干净收尾；静止桌面采集）；原始探针输出：
> `tinynav-sim/docs/oracle/`（频率以 bag 计数为准）。
> 用途：HIL looper_emu（hil-looper-emu-plan.md）的输出对拍 oracle——emu 发的
> 每一项都必须能在本表找到出处。

## 1. 契约话题表（emu 必须发布的全集）

| 话题 | 类型 | 发布 QoS | 实测频率 | frame_id | 内容要点 |
|---|---|---|---|---|---|
| `/camera/camera/infra1/image_rect_raw` | sensor_msgs/Image | RELIABLE, VOLATILE | **20.0Hz** | `camera_camera_left` | 544×640 **mono8**（w=544, h=640，竖幅） |
| `/camera/camera/depth/image_rect_raw` | sensor_msgs/Image | RELIABLE, VOLATILE | **12.8Hz** | `camera_camera_depth` | 544×640 **mono16**（毫米深度，16bit） |
| `/camera/camera/vio_image` | geometry_msgs/PoseStamped | RELIABLE, VOLATILE | **20.0Hz**（**逐帧发，非稀疏关键帧**；与 infra1 计数一一对应） | `world` | **T_world_camera**，与 depth/infra1 同 stamp（ExactTime 三元组） |
| `/camera/camera/vio_100hz` | geometry_msgs/PoseStamped | RELIABLE, VOLATILE | **98Hz** | `world` | 同一 T_world_camera 的高频重发 → 桥直通 `/slam/odometry` |
| `/camera/camera/vio_status` | std_msgs/String | RELIABLE, **TRANSIENT_LOCAL**（latched） | 1Hz | - | `TRACKING_STATIC` 等 VIO 状态字 |
| `/camera/camera/infra1/camera_info` | sensor_msgs/CameraInfo | RELIABLE | **20.0Hz** | `camera_camera_left` | 见 §3 内参；plumb_bob d=0（已整流），P=K |
| `/camera/camera/infra2/camera_info` | sensor_msgs/CameraInfo | RELIABLE | **20.0Hz** | `camera_camera_right` | 同 infra1 内参 |
| `/tf_static` | tf2_msgs/TFMessage | RELIABLE, TRANSIENT_LOCAL | 一帧 latched | - | 见 §4，逐字节照抄 |

（订阅侧的 QoS 来自 looper_bridge：sensor_qos=RELIABLE depth50；pose_sub 无显式
QoS=默认；tf_static_sub=TRANSIENT_LOCAL depth1。VOLATILE pub ↔ TRANSIENT_LOCAL
sub 的 vio_status 是 pub 侧自己 latched——emu 照抄。）

## 2. 同步与节拍（对 emu 的硬要求）

- **三元组 ExactTime**：looper_bridge 用 `message_filters.TimeSynchronizer`
  （严格相等，非 Approximate）配 (depth, vio_image, infra1)，queue 20。
  → emu 必须让三者 **stamp 完全一致**（同一时刻值），否则 0 命中。
  真机节拍：图像/vio_image 20Hz、depth 12.8Hz（depth 是图像 stamp 序列的子集，
  桥按 depth 节拍命中）。**emu 简化：三元组同频同 stamp 直发（如 15-20Hz）**，
  命中率 100%，节拍语义不变。
- **keyframe 判选在 bridge 内部**（`should_add_keyframe`：平移 ≥0.03m、旋转 ≥1°、
  距上帧 >3.0s，argparse 默认值）——emu **不需要**任何关键帧逻辑，逐帧发即可。
  （这比原计划 §2.2 更简单：删掉 emu 侧 keyframe 判选。）
- **stamp 是 looper 内部时钟**（开机相对秒，实测 sec≈10105，非 epoch；
  tf_static 的 stamp 是固定 epoch 1756222151≈固件时刻）。绝对值无消费者——
  navcore 不做绝对时间推理。emu 用 x86 墙钟、保持单调即可。
- 静止时 vio_image 与 vio_100hz 数值一致（同一 T_world_camera）。

## 3. 内参（emu 的 camera_info / gsplat 渲染面按此对齐）

```
width=544 height=640（注意：竖幅，gzsim 面的 544×480/272 需改）
fx=fy=302.77899169921875
cx=267.5927429199219  cy=320.5477294921875
distortion_model=plumb_bob  d=[0,0,0,0,0]  r=I  P=[[fx,0,cx,0],[0,fy,cy,0],[0,0,1,0]]
binning=0  roi 全零 do_rectify=false
```

## 4. /tf_static（5 条，emu 逐字节照抄）

```
camera_camera_left  → camera_camera_depth       t=(0,0,0)                 R=I
camera_camera_left  → camera_camera_imu_optical t=(0,0,0)                 R=I
camera_camera_left  → camera_camera_right       t=(0.0996236577630043,0,0) R=I   # 99.6mm 基线
camera_camera_left  → camera_camera_rgb         t=(0.0507298048449633,-0.0009218509497213958,0.0004383270107799681)
                                                R=(x=-0.0037471051339531022, y=-0.001724894113248995, z=0.0005698719111198014, w=0.9999913295571211)
camera_camera_imu   → camera_camera_left        t=(-0.03977611170763239,-0.02518247779983906,0.027494611044653186) R=I
```

## 5. 非契约但存在的 face 输出（emu 可不发）

`/camera/camera/imu`（BEST_EFFORT，实测 ~393Hz，rig 上 0 消费者）、`color/image_raw|rect_raw`
compressed、`infra1/2/image_raw`（未整流原图）、color camera_info。gsplat 面已有
color/imu 输出，保留无害；imu 若发建议对齐 BEST_EFFORT。

## 6. 节点身份

- looper 节点名：**`/insight_full`**（pilot `sensor_source_ready()` 的就绪判据，
  emu 节点名硬要求同此）。
- 桥消费后输出（navcore 侧）：`/slam/{odometry,odometry_visual,depth,
  disparity_vis,camera_info,keyframe_odom,keyframe_image,keyframe_depth}` +
  infra2/camera_info 转发——emu 不涉及。

## 7. emu 位姿语义补充（GT → 契约）

真 looper 的 `world` 原点 = VIO 初始化点，camera 指 left 光学系。
emu：`T_world_camera = T_world_base(gt_pose) × T_base_camera(挂载外参)`。
挂载外参是 emu 侧自由参数（真实 rig 上 looper 装狗头的位姿未采样）——
起步建议 (0.30, 0, 0.35) 米 + 前倾 ~10°，做成可配。
