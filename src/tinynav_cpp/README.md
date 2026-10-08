# src/tinynav_cpp — C++ 栈本体

单进程 rclcpp 组件宿主（进程内零拷贝），四个组件对应移植的 Python 节点
（perception / imu_propagator / mapping / planning）。

## 布局

```
src/core/        纯数学（SE3、PnP、IMU 积分）——禁止 include ROS 头
src/kernels/     raycast / 位姿图 / BA 内核（原 pybind 下沉）
src/planning/    占据栅格、ESDF、轨迹库（6 个 njit 函数的移植）
src/mapping/     VLAD、fusion window、A*、路径先验、live capture
src/trt/         TensorRT C++ 封装（8 引擎 + CUDA Graph）
src/components/  四个 rclcpp 组件 + main.cpp 单进程宿主
```

core/kernels/planning/mapping/trt 是纯库、gtest 可离线跑；ROS 只活在
components。移植函数保留 Python 命名并在文件头带
`// Port of reference/tinynav/core/<file>::<func>` 注释。

## 构建（只在容器里）

```bash
source /opt/ros/humble/setup.bash && colcon build --packages-select tinynav_cpp
```

**切换容器后必须重跑**：CMake 缓存带旧绝对路径。

## 测试

```bash
./build/tinynav_cpp/tinynav_core_test    # 72 用例；TRT 对拍探针在 tools/probes/
```

## 带图导航 / 回放 / 发目标速查

```bash
# bag 回放（无仿真器；/clock 由 bag 提供，yaml 设 use_sim_time:true）
ros2 launch tinynav_cpp tinynav.launch.py params_file:=<yaml>
ros2 bag play <bag> --clock

# 发导航目标（cpp 栈直发）
ros2 topic pub -w 1 -r 5 -t 5 /control/target_pose nav_msgs/msg/Odometry \
  "{header: {frame_id: world}, pose: {pose: {position: {x: 10.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}}"
```

建图 / reloc 排查的完整操作手册在 `docs/sim-mapping-runbook.md`；仓库根
`tools/` 是配套侧工具（build_map_live、export_map_v2、grab_topic、probes）。
