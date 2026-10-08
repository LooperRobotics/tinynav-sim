# Docker Compose

只使用一个常驻开发容器。

```bash
cp .env.example .env
docker compose up -d
docker exec -it tinynav-sim bash
```

若宿主用户或显卡设备组不是示例值，按 `id` 和 `getent group render` 修改 `.env`
中的 `HOST_GID`、`VIDEO_GID`、`RENDER_GID`。

进入容器后直接使用：

```bash
colcon build
./gazebo/run_simulator.sh --stack cpp --robot go2 --world gazebo/worlds/yard.sdf
```

容器已经配置 ROS 2 Humble、NVIDIA GPU、宿主显示器、TensorRT models、宿主
UID/GID，以及 `.container/` 下的非 root 构建目录。

停止：

```bash
docker compose down
```

## 纯仿真容器（sim service）

`docker/sim.Dockerfile` 构建只含仿真的镜像：gzsim（Ignition Fortress）+ gsplat
（3DGS，torch cu128 + motrixsim）+ 单机 CycloneDDS 配置。**不含** tinynav 栈
（无 TRT、无 /opt/venv、无 GTSAM）——栈跑在别的容器，通过 DDS 与仿真对话。

```bash
docker compose build sim          # 需要 pypi.motphys.com 可达（唯一非阿里源依赖）
docker compose up -d sim
docker exec -it gsim bash
# 容器内：
bash gsplat/run_gsplat.sh --scene church     # 或 bash gazebo/run_simulator.sh ...
```

约定：

- 镜像默认 `CUDA_ARCH=12.0`（RTX 5070），arch 同时用于构建期预编译和运行期
  JIT 缓存键（两者必须一致，否则每次启动重编 77 秒）。换 GPU 用
  `--build-arg CUDA_ARCH="8.9"` 重建；容器内手动 `TORCH_CUDA_ARCH_LIST` 覆盖可
  触发重新 JIT；
- DDS：镜像烤入 `/opt/dds/cyclone_localhost_unicast.xml`（单机 loopback 单播，
  对 clash TUN 劫持免疫，见 `docs/plan-sim-image-split.md`）。**同机双容器
  （模式 B）两边都必须 host network**，否则 127.0.0.1 的 Cyclone peer 对不上；
  另一侧容器挂仓库里的 `tools/probes/cyclone_localhost_unicast.xml` 即可对齐；
- 跨主机模式（USB 链路等）自行传 `CYCLONEDDS_URI` 覆盖烤入配置；
  `tools/preflight_dds.sh` 会在 run 脚本里拦住 TUN 开启时的必死组合。

## mjsim 镜像（MuJoCo 整体仿真）

`docker/mujoco.Dockerfile` 构建整体仿真镜像 `mjsim:jazzy`：MuJoCo 狗 +
wgpu splat（gausscam，PyPI 钉版）+ rclpy 契约面，跑 `mujoco/hil.py` 全家
（hil/sim/mj 三形态，默认 hil）。基于 ros:jazzy-ros-base（wgpu 0.32 要 py≥3.11，
humble 的 py3.10 不行），python 栈装在 `/opt/mjsim` venv（uv 安装，
`--system-site-packages` 保住 debian rclpy）。**不含** torch/TensorRT/gz。

```bash
# 构建前：model.zip 解压到仓根（生成 model/ —— splat 场景 + go2 网格，
# 均不入 git；布局见 model/VERSION.txt）
docker build -f docker/mujoco.Dockerfile -t mjsim:jazzy .
bash docker/run-mujoco-sim.sh [hil|sim|mj]                  # 默认 hil；GPU/X11//dev/input/host-net 全套
docker logs -f mjsim-hil                                    # 健康线：VIEWER mapped + face up
```

构建机只需要两样：本仓 + `model.zip`（解压出 `model/`）。gausscam 直接从
PyPI 镜像源装（钉 `gausscam==0.1.4`），不再带 wheel 文件；go2 网格与 splat
场景由 `COPY model /opt/model` 烤入镜像，运行期零外部依赖。

- **gausscam 版本**：PyPI 钉版（`gausscam==0.1.4`），hil 启动即走其完整
  API 面（Adapter/Pipeline/GaussianCloud/_K），版本坏了镜像当场起不来；
  升级 = 改 pin 重建。
- **ENV 钉死**（镜像层，均为实测踩坑产物）：`OPENBLAS_NUM_THREADS=1
  OMP_NUM_THREADS=1`（numpy OpenBLAS 池自旋，10.5→0.9ms-cpu/tick）、
  `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`（jazzy 默认 Fast DDS 发现/日志线程
  烧核且与栈的 Cyclone 不一致）、`MUJOCO_GL=egl`、`__GL_SYNC_TO_VBLANK=0`
  （vsync 会让 sync 阻塞 swap，给 200Hz 控制环引入 16ms 抖动）。
- **DRI 透传**：run 脚本对 4 个 DRI 节点逐个存在性判断（单卡机兼容，
  远端 2060M 只有一个 renderD 节点）。
- **viser 客户端 vendored 重编**（`mujoco/viser_client/`，见其 README）：viser
  客户端把方向键绑成相机旋转，**无官方开关**（上游 issue #259 open）——与全局
  /dev/input 遥操冲突（无视浏览器焦点）。客户端源码（v1.1.1，Apache-2.0）
  直接 vendor 在仓里、源码级删掉 5 个键位 + 5 行处理器；Dockerfile 用 node
  构建阶段（npmmirror）跑 viser 自己的 vite 管线产出客户端，COPY 进运行层并
  解码断言拦回归。python 侧 `viser==1.1.1` 与 vendored 客户端钉死同版本（ws
  协议耦合）。**坑**：wheel 的 `build/index.html` 是自解压壳（真身 base88+
  zstd 编码在 `data-p` 属性里），明文 grep 键位全是假阴性，改载荷的手术还曾
  引入 JS 语法错误——所以 vendor 源码从头编译。W/S/A/D、KeyQ 保留给相机。
- **资产在镜像里**：mujoco 代码 + `model/` 资产包（`COPY mujoco` +
  `COPY model` → 镜像内仓根 `/workspace/tinynav-sim/model`，与宿主机解压
  `model.zip` 后的布局完全一致）一起烤入，run 脚本不挂载仓库——克隆
  机器上只需本仓 + `model.zip`（解压到仓根）即可构建运行；gausscam 从
  PyPI 镜像源钉版安装，无 wheel 文件要带。
- 双机性能（viewer@10Hz 活动画面的整机 CPU / render 每帧）：

  | 机器 | 站桩 | 行走 | render/帧 |
  |---|---|---|---|
  | 本机 5070+ARL | 81-90% | ~同 | 28-43ms |
  | 远端 2060M | 63-66% | 54-65% | 66-77ms |

  全部零丢帧、cam 10Hz；跨厂 GPU 兼容矩阵（含 Intel ARL 降档）见
  `mujoco/README.md`。

