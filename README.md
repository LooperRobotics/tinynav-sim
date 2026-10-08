# tinynav-sim

tinynav 的仿真测试台。三套仿真互不依赖，各有独立文档，启动都是一两行：

## gazebo（gzsim）

宇树 go2 全链路（URDF + 步态链），worlds / 剧本场景齐全。

```bash
docker compose up -d && docker exec -it tinynav-sim bash   # 或进常驻 rig 容器
bash gazebo/run_simulator.sh                               # --stack/--world/--map 见 gazebo/README.md
```

详见 [gazebo/README.md](gazebo/README.md)。

## gsplat（3DGS）

MotrixSim 物理 + gsplat 渲染 + RL 步行策略，传感器面与 gazebo 逐字节一致。

```bash
bash gsplat/setup_env.sh                 # 宿主机一次性 bootstrap（venv + CUDA 12.8）
bash gsplat/run_gsplat.sh                # 容器内；--map 带图导航
```

详见 [gsplat/README.md](gsplat/README.md)。

## mujoco（mjsim 整体仿真）

MuJoCo 物理 + wgpu 3DGS 传感器视图：策略驱动 + HIL DDS 契约面 + viser 网页
可视化，独立镜像（代码与 model/ 资产包烤入镜像，运行期不挂载仓库）。

```bash
docker build -f docker/mujoco.Dockerfile -t mjsim:jazzy .   # 先解压 model.zip 到仓根
bash docker/run-mujoco-sim.sh                               # 默认 hil + viser :8012；--scene map3|map2
```

详见 [mujoco/README.md](mujoco/README.md) 与 [docker/README.md](docker/README.md)。

## 文档索引

| 文档 | 内容 |
|---|---|
| [gazebo/README.md](gazebo/README.md) | gz 仿真层：启动方式、DDS/分机、楼梯世界 |
| [gsplat/README.md](gsplat/README.md) | 3DGS 仿真层：架构、bootstrap、资产包、分机、坑 |
| [mujoco/README.md](mujoco/README.md) | mjsim：形态/场景/键盘/双专家、依赖与资产来源 |
| [docker/README.md](docker/README.md) | 容器与镜像：compose、纯仿真 sim 服务、mjsim 镜像 |
| [docs/sim-mapping-runbook.md](docs/sim-mapping-runbook.md) | 建图操作手册（在线建图、reloc 排查） |
| docs/gsplat-sim-progress.md | gsplat 集成进度与坑 |
| docs/stairs-gait.md · docs/migration-progress.md | 存档（历史） |
