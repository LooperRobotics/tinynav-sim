# Docker Compose

只使用一个常驻开发容器。

```bash
cp .env.example .env
docker compose up -d
docker exec -it tinynav-sim bash
```

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
