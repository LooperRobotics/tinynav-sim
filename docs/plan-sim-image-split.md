# 纯仿真镜像拆分：验证记录与设计方案

状态：**设计已拍板，纯仿真环境已落地**（拍板记录见 §8）。tinynav 栈侧完全
推迟——后续在其它容器里单独验证，本仓库只交付仿真环境。
探针容器 `gsplat-probe`（ros:humble-ros-base，环境齐全）保留在宿主机上可直接复用。

## 1. 目标

把仿真从 tinynav 全家桶镜像里拆出来：

- **容器① `tinynav-sim`（新镜像，不基于 tinynav）**：纯仿真 = gzsim + gsplat(3DGS)，体积最小化；
- **tinynav 栈**二选一：**模式 A** 跑在 USB 直连的 NavCore 工控机（同此前的分机部署）；**模式 B** 跑在 x86 宿主的另一个容器里。

## 2. 现状盘点（拆分前）

- 仓库 `Dockerfile` 是薄封装（`FROM uniflexai/tinynav@sha256:96f8…`，43.5GB 全量镜像）；
  `compose.yaml` 只有一个 service 且用 `image:` 直接引用基础镜像（不走 `build:`）。
- 日常实际用的是**仓库之外**的常驻 rig 容器 `tinynav`（`tinynav-runtime:x86_64`，47.8GB，
  仓库挂 `/workspace/dm/tinynav-sim`）。
- 仿真面真正依赖（从 tinynav 镜像里用到的全部东西）：
  - gzsim 面：ROS2 Humble（ros_gz bridge、xacro、robot_state_publisher、ros2_control）+
    Ignition Fortress（渲染走 GPU OpenGL，**不需要 CUDA**）+ python3/rclpy/numpy/scipy/cv2
    （全部跑仓库内的 `reference/tinynav/platforms` 和 `gazebo/scene/*.py`）；
  - gsplat 面：`/opt/venv_gs`（torch cu128 + gsplat + motrixsim + onnxruntime，~7GB）+
    `/opt/cuda-12.8` nvcc（~1.3GB，RTX 5070 的 sm_120 必需）+ rclpy 桥 + `GS_PLAYGROUND_ROOT` 挂载资产；
  - **不碰**：TRT、`/opt/venv`、GTSAM、tinynav_cpp。
- 拆镜像必须处理的硬依赖坑（实施清单见 §7.5）。

## 3. 验证一：纯 gsplat 在 humble 裸镜像运行（✅ 通过）

容器：`docker run --name gsplat-probe ros:humble-ros-base`（host network + --gpus all +
挂载本仓库 + gs_playground checkout），bootstrap 脚本 `tools/probes/gsplat_humble_bootstrap.sh`
（可复现，已入库）。全程 uv + 阿里源：

| 包 | 源 | 备注 |
|---|---|---|
| torch 2.7.0+cu128 | `mirrors.aliyun.com/pytorch-wheels/cu128`（flat 页，用 `--find-links`） | nvidia 依赖走阿里 pypi |
| motrixsim-core==0.7.1.dev97295 | `pypi.motphys.com`（私有） | **必须加 `--index-strategy unsafe-best-match`**——阿里镜像也收录了该包名，uv 默认 first-match 策略会拒绝去私有源找 dev 版本（`gsplat/setup_env.sh` 的原始命令在新版 uv 上直接失败） |
| gsplat 1.5.3 / gaussian_renderer 0.2.0 / numpy 2.2.6 等 | 阿里 pypi | 与 setup_env.sh 的 pin 一致 |

结果（全部容器内真实完成）：

- gsplat CUDA kernel **JIT 编译 73.6 秒**（非拷贝预编译产物）；
- `torch 2.7.0+cu128, CUDA available, RTX 5070`；
- `sensor_server.py --out` 全链路：教堂 766 万 + 机器人 60 万高斯加载、61 帧相机 @15.2Hz、
  801 帧 IMU @200.2Hz、9 张 PNG 正常落盘（4s sim / 9.9s wall，rtf 0.40——rig 记录 0.91，
  差异归因宿主桌面负载，非容器开销，最终镜像需干净环境复测）。

**下载测速**（容器裸连 vs 走宿主代理 127.0.0.1:7897）：

| 路线 | 实测 |
|---|---|
| aliyun pypi / pytorch-wheels 裸连 | **35 MB/s** |
| pypi.motphys.com 裸连 | **26 MB/s** |
| 同上两路线走代理 | 反而降到 15–24 MB/s（代理只对国际流量有意义，国内镜像别走代理） |

顺带的两条宿主网络事实：`net.core.wmem_max/rmem_max = 4MB`（Cyclone 配置 socket
buffer 需求别超过 4MB，>4MB 会 rmw_create_node 直接失败）；Linux UDP 单报文上限
65507 字节（>64KB 的 sendto 立即 EMSGSIZE，与 buffer 无关——排查网络时别拿大报文当探针）。

## 4. 验证二：rclpy 与 venv_gs 单进程共存（✅ 通过）——回答"能不能不走共享内存"

**先澄清**：shm ring 从来不是跨容器方案（/dev/shm 不跨容器），它是容器内
sensor_server（venv_gs，numpy 2.x）与 gs_ros_bridge（系统 python3 + rclpy）两进程间的
通道，存在理由是 python 环境冲突 + 渲染循环与 DDS 节奏解耦。

**实测结论：环境冲突不存在，单进程 fused 直发 DDS 可行**——

- venv python 只要 `source /opt/ros/humble/setup.bash`（PYTHONPATH 带入 ROS 的
  site-packages）就能 import rclpy，与 torch/CUDA/gsplat/numpy 2.2.6 同进程共存，
  无需 `--system-site-packages`；
- 发布成本：261KB mono **0.45 ms/帧**、783KB rgb8（真实 color 帧）**0.78 ms/帧**，
  15Hz 预算 66.7ms 内绰绰有余；
- 跨容器（gsplat-probe → tinynav）：**14.4 Hz 持续送达**，零丢失语义下 ~96% 有效帧。

工程建议：拆容器先保留 bridge + ring 形态（坑已踩完、稳），fused 单进程化作为后续
独立优化（改动集中在 server/bridge 进程边界，与容器拆分正交）。

## 5. 重大环境发现：clash TUN 正在杀死本机所有 DDS（根因与判据）

排查中一度测出"publish 一有订阅者就阻塞 166ms/帧、订阅端 0 帧"，最终根因：

**宿主跑着 clash-verge/mihomo 的 TUN 模式**（`Meta` 接口 198.18.0.1/30 +
`ip rule 9002: not from all iif lo lookup 2022`），除 loopback 外的一切路由——
**包括 DDS 组播（源地址未定时 src=0.0.0.0 被规则命中）**——被甩进 TUN 黑洞。

症状矩阵（全部复现）：FastDDS/Cyclone 两种 RMW、3KB/261KB 消息、probe/tinynav
两个容器，订阅端 0–1 帧、发布阻塞。rig 容器同样中招——**这意味着该机现有的单机
仿真流程在 TUN 开启期间是跑不通的**（与镜像无关）。

判别实验链（以后排查 DDS 可用同一套路）：

1. 无订阅者时发布 261KB Image：**0.042 ms/帧** → rclpy 序列化无问题（memcpy 快路径）；
2. 任何尺寸 + 有订阅者即劣化 → 传输/发现层问题，非序列化；
3. `ip rule` 看到 `lookup 2022` 全流量劫持 → TUN 实锤。

**解法（已验证）**：Cyclone 钉 lo + 禁组播 + 显式 127.0.0.1 peer，配置已入库
`tools/probes/cyclone_localhost_unicast.xml`。同机多容器/多进程全通。

> 注意与 deploy.md §4.9 坑①（"接口表绝不列 127.0.0.1"）不矛盾：那条适用于
> 多宿主机器（lo 会抢走组播口导致参与者对全网不可见）；本配置禁用了组播、
> 只用于单机，语义完全不同。也别拿那条去"修"这份配置。

### 5.1 机理（完整因果链）与判决

1. TUN 的 catch-all 策略路由（`not from all iif lo lookup 2022` → 表 2022 默认
   经 Meta）把 main 表无具体路由的目的地全部导进用户态 clash——组播地址
   239.255.0.1 在 main 表没有路由，命中；
2. clash 是单播代理：组播包既不能也不该转发（link-local scope、代理层无"加组"
   概念），mihomo TUN 栈直接丢弃 → SPDP 发现静默消失；
3. 症状分两层——**发现层死**：参与者互相不可见（订阅端 0 帧）；**数据层半死**：
   已发现对端的单播数据包进 TUN 无人读，发送端 socket 缓冲堆满 → DDS 写阻塞。
   这解释了症状矩阵的全部特征：无订阅者 0.042 ms/帧（Cyclone 对无人听的数据
   不写 socket）vs 有订阅者 166 ms/帧（缓冲以帧速率堆满后背压节流）；
   FastDDS/Cyclone、3KB/261KB 全中招——问题在 IP 层，与 DDS 实现无关；
4. localhost 修复生效的原因：钉 lo + 禁组播 + 显式 127.0.0.1 peer 后，全部流量
   是 lo 上的单播 UDP，命中规则自带的 `iif lo` 例外，根本不进策略路由。

**判决：不是 DDS 的 bug，是三层责任叠加**——

| 层 | 评价 |
|---|---|
| DDS | "局域网组播可用"是 RTPS 规范的标准假设，被下一层破坏不是它的错；能批评的是可观测性（发现黑掉一声不吭，启动时探测组播可达性并告警算改进项而非 bug） |
| clash TUN | 规则集没豁免 224.0.0.0/4 是配置缺失（新版 mihomo auto-route 通常会加）；代理在架构上也不可能转发 link-local 组播——真正该"修"的一层 |
| UDP/组播语义 | 决定了必然静默失败，没有任何一层有义务报错 |

同类案例：VPN TUN 杀 mDNS/AirPlay/DLNA，机理一致。修复矩阵按层次：网络层
（clash 加组播豁免或跑仿真关 TUN）、DDS 层（钉 lo 单播，已烤进镜像）、应用层
（preflight，见 §7.6）。

## 6. 依赖源码形态（黑盒边界）

| 包 | 形态 | 源码 |
|---|---|---|
| gsplat 1.5.3 | 完整 py + CUDA C++ 源码，.so 为首次 import 时本地 JIT | ✅ 开源（nerfstudio-project/gsplat） |
| gaussian_renderer 0.2.0 | 纯 Python（13 个 .py，无 .so），公有 PyPI 可拉 | ✅ 全在 site-packages（Motphys 配套包，gs_playground checkout 不内嵌） |
| motrixsim-core 0.7.1.dev97295 | **闭源 .so** + 8 个 .py 包装，私有索引发行 | ❌ 黑盒，pin 死版本；镜像构建机必须可达 pypi.motphys.com（阿里源覆盖不到的唯一依赖） |

推论：渲染链可审计可 patch（改完重 JIT）；motrixsim 换版本需回归。

## 7. 设计方案

### 7.1 `docker/sim.Dockerfile`（gsplat 侧已验证，gzsim 侧照 go2_sim 配方）

- `FROM ros:humble-ros-base` + apt：ignition-fortress、ros-humble-ros-gz、
  ros2-control、gz-ros2-control、xacro、robot-state-publisher、rviz2、
  python3-numpy/scipy/opencv/pil、python3-pynput、mesa（llvmpipe 给 gsplat viewer/gz 兜底）、
  rmw-cyclonedds-cpp；
- uv 装 venv_gs（阿里源 + motphys + `unsafe-best-match`）+ micromamba 装 CUDA 12.8 nvcc
  （最终镜像要网络安装保证可复现，不能学探针从宿主 `/home/dm/.local/cuda-12.8` docker cp）；
- 构建期 precompile gsplat 内核（`TORCH_CUDA_ARCH_LIST=12.0`，先把 5070 写死）；
- 烤入 `cyclone_localhost_unicast.xml`；**不含** TRT/tinynav_cpp//opt/venv/GTSAM。
- **实测落地 17.7GB**（venv_gs ~7GB 是大头，fortress 全套 + CUDA 12.8 nvcc
  1.3GB + 基座 1.17GB）；容器可写层 ~0.4MB ≈ 零——一切来自镜像层 + 挂载，
  这正是拆分要的性质。纯 gzsim 变体可压到 ~6GB（go2_sim 已证），拍板不单做（§8.2）。
- 构建期三个实测发现（都写进了 Dockerfile 注释）：① aliyun 镜像对 apt 客户端
  限速 ~300KB/s（curl 同 host 28MB/s、强制 IPv4/host network 均无效）→ apt 用
  默认国际源走 clash（update ~10s），uv/pip 才走 aliyun（35MB/s）；
  ② buildkit `--network host` 需 `--allow=network.host` 才生效（compose 对应
  `build.network: host`）；③ gsplat JIT 缓存键绑定 `TORCH_CUDA_ARCH_LIST`，
  build-arg 的 CUDA_ARCH 必须同时烤入运行期 ENV，否则每个新容器首启重编 77s。

### 7.2 DDS 三模式（sim.launch.py 增加 `dds:=local`）

| 模式 | 场景 | 配置 |
|---|---|---|
| `local`（新增） | 模式 B：同机双容器 | `cyclone_localhost_unicast.xml`（§5，已实测） |
| `cyclone`（现有） | 模式 A：USB 链路，x86 为 host 角色、组播可用 | `cyclonedds_x86.xml`（NIC 名需参数化，现写死 enxaac4cf20a190） |
| hil MTU 化（现有 `gsplat/configs/hil_cyclone_x86.xml`） | 模式 A：gadget 角色链路（UDP 突发丢 ~5%，deploy.md §4.9 铁证） | FragmentSize 1344 / MaxMessageSize 1456 / 重传单片一报；与 NavCore 侧 FastDDS 跨厂商 RTPS 互通已实测 |

### 7.3 compose（sim service 已落地；stack service 推迟）

- `sim` service：新镜像；host network、`--gpus all`、X11、`/dev/dri`、
  `GS_PLAYGROUND_ROOT`、`FACTORY_MODEL_ROOT`；
- `stack` service：uniflexai/tinynav 基础镜像（或 tinynav-runtime），挂 `TINYNAV_MODELS_DIR`，
  不需要 X11；模式 B 下跑 `tinynav.launch.py`（同机无需 bridge/relay）+ localhost cyclone 配置；
- `navcore` profile：不起 stack，sim 带 USB 模式 DDS 环境变量。

### 7.4 NavCore 两子模式（整体推迟，待 NavCore 信息）

- **A1**：NavCore 跑本仓 C++ 栈（`orin_stack.launch.py`，同 Orin 演练，NavCore 只是另一个
  Cyclone peer）——需要 NavCore 是 x86_64 + Ubuntu 22.04，且 tinynav_cpp 在它上面构建；
- **A2**：NavCore 跑生产 python 栈——perception（TRT）必须留在 x86 stack 容器，拓扑变成
  "sim + perception 容器 ↔ NavCore（bridge+三件套）"，多一跳。

### 7.5 脚本小手术清单（1/2/3/5 与 4 的 `dds:=local` 已落地；USB NIC 名
### 参数化随模式 A 推迟）

1. `gazebo/run_simulator.sh:287` rviz 写死镜像内 `/tinynav/docs/vis.rviz` → 改指仓库
   `docs/vis.rviz`（两者内容已分叉，md5 不同，需对齐）；
2. `gazebo/dog_state.sh:122`、`gsplat/gs_state.sh:45,123` 写死 `/opt/venv/bin/python3`
   （只需 numpy）→ 普通 python3；
3. `gsplat/setup_env.sh` motrixsim 行补 `--index-strategy unsafe-best-match`；
4. `sim.launch.py` 增加 `dds:=local`；USB 模式 NIC 名参数化；
5. run 脚本的 `CYCLONEDDS_URI` 清理逻辑按模式区分（local 模式不再清空而是指向 localhost 配置）。

### 7.6 TUN 处理（模式 A 前置；模式 B 不需要）

拍板：preflight 检查 + 报错（`tools/preflight_dds.sh`，已接入两个 run 脚本；
loopback 单播配置激活时放行并提示，TUN 开启 + 物理网卡 DDS 的必死组合直接
拒绝启动）。永久 ip rule 放行不做；跨主机模式跑仿真期间关 TUN。

### 7.7 验收计划

镜像构建后：gz headless 冒烟（camera 话题 hz）→ gsplat 冒烟（sensor_server --out）→
模式 B 全链路（sim + stack 容器发目标走一圈）→ 模式 A 待 NavCore 信息齐后排期。

## 8. 拍板记录

1. **落地顺序**：只做纯仿真环境（gzsim + gsplat 镜像、DDS local 接口、
   preflight、compose sim service）；tinynav 栈与模式 A（NavCore）整体推迟，
   栈的验证后续在其它容器里单独做。镜像内不含栈的任何部分。
2. **镜像形态**：一个镜像双面（~13–15GB），纯 gzsim 变体不单做。基座
   `ros:humble-ros-base`，venv_gs + CUDA nvcc 网络安装烤入（构建期 motphys
   可达）；`CUDA_ARCH` 做成 build-arg 并烤入运行期 ENV（默认 12.0 = RTX 5070；
   JIT 缓存键绑定该值，换 GPU 用 build-arg 重建镜像）。
3. **TUN**：preflight 检查 + 报错（见 §7.6），不做永久 ip rule。
4. **DDS 接口契约**（给未来的栈容器）：镜像烤入 loopback 单播 Cyclone 配置；
   同机双容器两边都必须 host network；另一侧挂仓库
   `tools/probes/cyclone_localhost_unicast.xml`；跨主机模式自行覆盖
   `CYCLONEDDS_URI`。已写进 `docker/README.md`。
5. **接口属性写法统一（name=）与 USB NIC 名参数化**：拍板按 `name=` 统一，
   实施随模式 A 推迟；`hil_cyclone_x86.xml` 的 `address=` 疑似静默忽略过，
   届时一并改并在 USB 链路复测。

## 9. 产物清单

- 探针容器 `gsplat-probe`（ros:humble-ros-base + 全套已验证环境；容器内 /tmp 的
  配置与脚本重启即失，以仓库为准）；
- `tools/probes/gsplat_humble_bootstrap.sh` —— 可复现 bootstrap（uv 全阿里源）；
- `tools/probes/gsplat_fused_pub.py` / `gsplat_fused_pub2.py`（--rate/--seconds/--heavy/--rgb
  参数化）/ `gsplat_fused_sub.py` / `gsplat_fused_sub_rate.py` —— DDS 发布成本/速率探针；
- `tools/probes/cyclone_localhost_unicast.xml` —— 单机 Cyclone 配置（§5，已实测）。
- `docker/sim.Dockerfile` —— 纯仿真镜像（gzsim + gsplat 双面，CUDA_ARCH 参数化）；
- `tools/preflight_dds.sh` —— DDS preflight（TUN 必死组合拦截，两个 run 脚本接入）；
- `compose.yaml` sim service —— 纯仿真容器（gsim，host network + GPU + X11）；
- 脚本手术 §7.5（rviz 路径、numpy python 回退、setup_env flag、dds:=local、
  run 脚本 URI 保留逻辑）。
