# SigLIP2 语义建图 + 地图格式演进 实现计划

> 状态：**全部完成**——阶段 0/1/2/3/4 全部完成并验证
> 关联：docs/migration-progress.md（格式纪律）；知识库 dm-tinynav qa 1.10/1.11/1.12（建图链路线索+实测）

## 阶段 0 完成记录

### 0.1 完成
- 双塔 ONNX 重导（fp32 权重 + fp32 L2 尾部；model.half() 方案被 onnxruntime 拒载，弃用记录见 qa 1.11）
- plan：`model_test/siglip_cmp/siglip2/siglip2_base_p16_224_{image,text}_fp16_x86_64.plan`（trtexec --fp16，Makefile 配方）
- 验收 PASS：image 范数 0.999827 / text 0.999898；权重零差（head.probe/token_embedding max|diff|=0.0）
- v1 text plan 0 字节事故已修复（tinynav-pilot models 下重建，223MB）

### 0.2 完成
- 地图：tinynav_sim_db/maps/map_2026_09_16_13_45_23（46 关键帧，rgb_images_db 有 VideoDB）
- 查询：8 条（7 中文 + 1 英文对照），GT 人工看图标注（`model_test/siglip_cmp/queries.json`）
- 结果：v1_stored top1/top5 = 0.625/0.750；v1_fresh 0.625/0.625；**siglip2 0.750/0.875 ≥ 基线 → 定案 SigLIP2**
- 脚本：`model_test/siglip_cmp/bench_retrieval.py`（可复跑）；明细 out/bench_retrieval.json

## 阶段 0/1/2 完成记录

### 0.3 完成
- `tools/mapio/writer.py`：MapWriter 抽象 + V2NpyWriter（sidecar：semantic_embeddings.npy N×768 f32 行序=pose_timestamps、零行=缺失 + semantic_meta.json 真 schema）
- export_map_v2.py 重构为 V2NpyWriter 前端；与旧导出器逐字节一致（16 个 npy 全部 IDENTICAL）
- C++ 老 reader（load_map_v2）对带/不带 sidecar 的目录均 LOAD OK（tools/probes/probe_map_v2 实测）

### builder 本体（tools/build_map/，阶段 1）
- `builder_node.py`：reference BuildMapNode 语义（4 话题 ApproximateTime 同步 / DINOv2 全局+patch / SP 特征 / DINO 回环 0.90 → LightGlue+PnP≥100inliers / pose graph pybind solve_pose_graph，在线 ratio 1.1 + 失败降级 WARN）+ color 超时守卫（20 关键帧无 color 即 WARN，reference 1.4 教训）+ SigLIP2 嵌入（0.5/0.5 预处理，plan 烘 L2，范数运行时断言 [0.99,1.01]）
- `run_bag_build.py` 编排器：bag 话题扫描 → 无 /slam 自动拉 perception 子进程（**必须 source /3rdparty/message_filters_ws/install/local_setup.bash**，否则 InputAligner import 失败——run_simulator.sh:149 同款）→ BagPlayer + builder 单线程 executor → finalize
- `siglip2_engine.py`：SigLIP2 image plan 封装

### 首跑验证（bag_2026_09_16_13_43_18，4 分钟端到端，阶段 1.5）
- 44 关键帧全部带 SigLIP2 语义嵌入；VLAD (44,24576) 单位范数；occupancy+SDF 生成正常
- **与 pilot v1 地图对拍**：41 共享时间戳位姿位置误差 mean 2.8cm / max 4.0cm ✓
- **C++ 加载**：probe_map_v2 load_map_v2 带/不带 sidecar 均 OK ✓

### 阶段 2：v2→v1 反向桥（tools/convert_v2_to_v1.py）
- poses.npy(pickled dict) / metadata / vlad_descriptors / features([1,N,2]/[1,N,256]/[1,N,1]) / depths(f32 米) / semantic_embeddings 全部 shelve 写回；主线消费面格式验证通过（/tmp/v1_bridged 抽查）
- 不产出：图像 VideoDB、DINO 全局嵌入、patch_tokens（v2 无此数据）；retrieval 排名可用但结果图保存需要图像

## 已拍板的决策

1. **路线 B**：tinynav-sim 内独立 python builder，输入 rosbag，输出 v2 地图 + SigLIP2 语义嵌入
   （动机：sim 自包含、用未建过图的 bag 测试、不依赖 pilot v1 工具链）
2. **SigLIP2 归一化烘进模型**：重导 ONNX（vision+text 双塔补 L2Normalize，text 顺带修 fp16）
3. **语义嵌入键 = 关键帧时间戳 ns**；模型名/版本入 sidecar，从第一天建真 schema
4. **benchmark 先行**：验收阈值 = SigLIP2 检索命中率 ≥ v1 基线
5. **SQLite 为地图格式 v3 目标**（省空间/省开销/多语言三标准下的选型 winner）
6. **builder 内格式藏在 MapWriter 接口后**（半天成本，换取将来 v2→SQLite 零改动）

## 阶段 0：前置基建（~2 天）

### 0.1 SigLIP2 模型重导（半天）
- 改 `model_test/siglip_cmp/export_siglip2_onnx.py`：双塔输出后接 `x / x.norm(p=2, dim=-1, keepdim=True)`
  （激活保持 fp32 做归一化，权重仍 fp16）；text 塔导出时转 fp16（修 1.13GB 问题）
- tinynav 容器内 trtexec 重建 plan（Makefile 同款配方），命名 `siglip2_base_p16_224_{image,text}_fp16_{arch}.plan`
- **验证**：plan 输出范数 ∈ [0.99, 1.01]；与 v1 引擎同图嵌入 cosine 相似度记录基线值
- 顺带重建现役 v1 text plan（现 0 字节/TRT10.13 反序列化失败）
- 风险：容器内 TRT 版本兼容性（有先例），先冒烟再批量

### 0.2 benchmark 三件套（1 天）
- 选定一个含 rgb_images_db 的现成 pilot 地图（**待确认用哪个**）
- 查询集：5-10 条中文文本查询（覆盖典型目标：物体/区域/功能点），人工标注应命中关键帧
- v1 基线：v1 text plan × 地图 semantic_embeddings.db → top-1/top-5 命中率
- SigLIP2 候选：同批 RGB 过新 image plan 重算嵌入 → 同检索
- **产出**：验收阈值结论（≥ 基线 → 定案 SigLIP2；明显差 → 回退讨论）

### 0.3 MapWriter 接口 + v2-npy 实现（半天）
- `tools/mapio/writer.py`：`MapWriter` 抽象（add_keyframe/add_array/finalize/set_semantic_meta...）
- `V2NpyWriter`：对齐 export_map_v2.py 列式约定（u16 毫米深度、f32 VLAD、offsets 变长行、pose f64 扁平）
- sidecar：`semantic_embeddings.npy`（N×768 f32，行序=pose_timestamps）+
  `semantic_meta.json`（schema 定稿：`{model, model_version, normalized: true, key: "ts_ns", created, builder_version}`）
- **纪律**：numpy 回读校验 + 缺文件降级语义与 map_v2.hpp:68-70 契约一致

## 阶段 1：路线 B builder 本体（~3-4 天）

### 1.1 编排器（半天）
- bag → 拉起 reference 快照 `perception_node.py`（VIO 复用，不重写）+ 新 builder，进程组管理
- 话题/QoS 对齐核查（参考 run_rosbag_build_map.sh 拓扑）

### 1.2 builder 核心（1.5 天）
- 四话题 ApproximateTimeSynchronizer（keyframe_image/odom/depth + color），
  color 超时守卫：N 关键帧无 color 帧 → WARN（不静默，吸取 1.4 坑）
- DINOv2 全局嵌入 + SP 特征逐帧入库（内存暂存，close 时写盘）
- 回环：DINO cosine 候选（阈值沿用 0.90）→ LightGlue + PnP 验证
- VLAD streaming 词表训练（close 时）、occupancy raycast 烘焙

### 1.3 pose graph（0.5 天，风险已降低）
- 首选：import pybind 绑定 `tinynav_cpp_bind.pose_graph_solve`
- **前置确认项（预计可满足）**：AGENTS.md 实据——镜像以 namespace package 形式自带
  `tinynav.tinynav_cpp_bind`（.so），import 前应冒烟确认符号在
- fallback：先关闭全局优化跑通（maybe_run_global_refinement 条件触发，可跳过）

### 1.4 SigLIP2 嵌入接入（0.5 天）
- color 帧按时间戳对齐关键帧 → image plan（烘 L2）→ 直接得单位向量 → sidecar
- TRT python API（容器内 bench 脚本同款套路）

### 1.5 v2 写出 + 校验（0.5 天，复用 0.3）
- **验证锚点**：旧 bag 重建 → poses/VLAD 索引与 pilot v1 产物对拍（容差内一致）→
  sim C++ map_node 加载（含新 sidecar，老 reader 不炸）→ reloc 闭环通过

## 阶段 2：v2→v1 反向桥（~0.5 天）
- `tools/convert_v2_to_v1.py`：v2 npy → v1 shelve/pickle 布局（poses dict、各 .db）
- semantic_embeddings.db 一并写回 → 主线 `retrieval_map_by_language.py` 直接可用
- **价值闭环**：sim/C++ 产物可进主线 3DGS（convert_to_colmap）/检索工具链

## 阶段 3 完成记录

### 3.1-3.4 完成（SQLite v3）
- **schema 定稿**（`map.sqlite`，WAL，`PRAGMA user_version=3`，meta.format=tinynav_map_v3）：
  `meta(k,v)`（含 blobs_meta JSON + 每个 blob 的 `blob.<name>.dtype/.shape/.json` 冗余键，C++ 侧零 JSON 依赖）/
  `keyframes(ts PK, pose BLOB 16×f64 行主序)` /
  `arrays(ts,name,dtype,shape,data, PK(ts,name))`（depth `<u2` mm、feature_kpts/feature_descps `<f4`、feature_mask `<u1`、vlad_descriptor `<f4`、semantic_embedding `<f4`；shape 存 JSON 数组文本；feature_offsets 不存，按行数重建）/
  `images(ts,kind,codec,data)`（schema 预留，builder 不落图）/
  `blobs(name PK, data)`（vlad_centres + intrinsics/occupancy/sdf/path_*/baseline/rgb_camera_intrinsics 等 aux + semantic_meta JSON 文本）。
- **python 层**：`tools/mapio/sqlite_writer.py`（SQLiteWriter = 第二个 MapWriter 实现；
  `load_map_v3()` reader；finalize 走 load_map_v3 全量回读校验 shape/dtype/字节）+
  `tools/migrate_map_to_v3.py`（源自动识别：pose_timestamps.npy→v2、poses.npy→v1[shelve 读取对齐 export_map_v2.py]）。
- **C++ 读取层**：`mapping/map_v3.{hpp,cpp}`——与 map_v2 同款消费面
  （timestamps/poses/vlad_centres/VladIndex + has_frame/get_features/get_depth；
  features/depth 零拷贝视图，sqlite BLOB 指针按行 keep-alive prepared statement，
  MapV3 所有者必须存活）+ `semantic_embeddings`(N×768，空矩阵=块缺失/全零) + `meta`。
  契约与 v2 一致：keyframes/vlad/depth/features 任一缺失 → false+error 调用方降级；
  语义缺失/全零 → 返回 true 语义留空。sqlite3 链进 tinynav_core（纯库，无 ROS 头）。
- **测试/探针**：`test/test_map_v3.cpp`（fixture=fixtures/map_v3/bag_13_43_18/map.sqlite，
  由迁移器从 output/map_build_siglip2/bag_13_43_18 生成；poses 1e-9、vlad/depth 位级、
  semantic 非零行数对拍；fixture 缺失 GTEST_SKIP 打印生成命令）+
  `tools/probes/probe_map_v3.cpp`。全量 74/74 绿。
- **验收**：python round-trip 逐数组 np.array_equal 全过（depth u2 / vlad f32 位级）；
  v1→v3、v2→v3 两条迁移路径均实测；probe_map_v3 LOAD OK 44kf；
  probe_map_v2 回归 OK；export_map_v2.py（V2NpyWriter）行为不变。
- 坑：① numpy `ascontiguousarray` 把 0-d 提升为 1-d（baseline.npy 是 0-d 标量，
  回读校验拦下）；② Eigen 默认列主序，pose BLOB memcpy 会转置（gtest 拦下，
  逐元素填充修复）；③ sqlite3_column_blob 指针只在下一次 step 前有效，
  全表扫描时早期行会悬垂 → 每行独立 keep-alive statement 才是真零拷贝。
- 未做（按计划后续单独立项）：mapping_component 接线 load_map_v3、LiveCapture 切 v3 写出。

## 阶段 4：E2E 验收（~1 天）
- 未建过图的新 bag（真实数据）→ builder → v2/v3 地图
- C++ 栈重定位闭环（0 失败锚点）+ benchmark 阈值复验 + v2→v1 桥进主线 3DGS 冒烟

### 4.1-4.4 完成记录

**4.1 未建过图的 bag → builder（通过）**：bag_13_42_27（raw 传感器 bag，无 /slam）→
36 关键帧全部带 SigLIP2 嵌入（范数 [0.99959,1.00041]）、VLAD 单位范数、
probe_map_v2 LOAD OK、v3 迁移 + probe_map_v3 LOAD OK。产出
`output/map_build_siglip2/bag_13_42_27{,_v3}/`。

**4.2 benchmark 阈值复验（通过）**：builder 在线嵌入与 0.2 离线 benchmark 质量一致
（范数同级；相邻帧 cos 0.989 > 随机对；跨地图 per-frame best-match cos 0.958-0.974；
8 条查询检索经 bag 原图抽帧目检 top-1 正确）。注意：跨轨迹移植 GT 要小心可移动物体假阴性。

**4.3 C++ 重定位闭环（通过，0 失败锚点证据完整）**：
- 判据链：`map v2 loaded ... relocalization enabled` → `reloc hit: inlier ratio 1.00`
  → Ceres `Initial 1.89e-01 → Final 5.59e-25 CONVERGENCE` → `/map/relocalization`
  位姿沿轨迹单调推进，末点与地图最后一帧吻合 <1cm；全程 0 个 reloc-failure dump。
- 标准做法（已验证，比拉 gz 全场轻）：**C++ 栈可直接消费 raw bag**——
  `ros2 bag play --clock -r 0.2`（背压：感知 ~100-140ms/帧）+ C++ 单进程栈
  `use_sim_time:=true map_path:=<v2图>`。
- 重要环境发现：gz 现场 reloc 对新 bag 地图必败的根因是**场景资产漂移**
  （factory_01 模型在录制后 3 小时被重出，meshes/textures 09-17 又更新；
  dump 现场画面与 bag 同位姿画面完全不同）——非 builder/地图/C++ 栈问题。
  以后"bag 建图→gz 现场 reloc"必须先核对模型 mtime。

**4.4 v2→v1 桥进主线（按预案降级，限制记录）**：
- 桥产物 11 类 v1 文件齐全；主线 retrieval 数据面（TinyNavDB + semantic 矩阵 +
  v1 text plan 排名 top-3）跑通。
- `convert_to_colmap_format.py` 在 tf-less bag 的地图上止步于 `T_rgb_to_infra1.npy`
  缺失（bag 无 /tf，builder 无法计算 rgb→infra1 外参），下一 blocker 是
  `rgb_images_db`（v2 无图）——结构性限制。若要主线 3DGS 真可用，方向是
  colmap 工具加"无图只导位姿/点云"降级模式，而不是给桥补数据。
- `retrieval_map_by_language` 在缺 DINO embeddings.db 时于图像保存前 KeyError
  （比预期"无图像 raise"更早），同为已知限制。

## 风险与开放项

| 风险 | 缓解 |
|---|---|
| 容器 TRT 版本与 plan 兼容性（text plan 有前科） | 0.1 先冒烟；失败则锁 TRT 版本重出 |
| pilot pybind 绑定容器内不可用 | 1.3 fallback 先跳过全局优化，并行确认 |
| v2 sidecar 加进地图目录后 C++ loader 行为 | 1.5 验证锚点必跑；map_v2.cpp 对未知文件应容忍（待核实） |
| 伪彩色 sim bag 的嵌入质量 | 已降优先级（后续用真实数据）；benchmark 用真彩色地图不受影响 |
| 外部场景资产（metaverse-source/factory_01）无版本管理，录制后重出即建图-现场不一致 | "bag 建图→gz 现场 reloc"前核对模型 mtime；reloc 回归改用 bag play 闭环（4.3 已验证） |
| 主线工具链依赖 v2 结构性缺失的数据（T_rgb_to_infra1/图像/DINO embeddings） | colmap 工具加"无图只导位姿/点云"降级模式；不在桥侧补数据（4.4 记录） |
| **待确认**：benchmark 用哪个 pilot 地图 | 用户指定 |

## 时间线

```
第 1 周: [0.1]→[0.2]→[0.3]→[1.1][1.2]
第 2 周: [1.3][1.4][1.5]→[2]→(并行)[3.1-3.4]→[4]
```

## 附：本次讨论的设计结论（防漂移）

- v1 = Python 对象图快照（pickle/shelve/VideoDB），C++ 不可读写是结构必然，非实现懒惰
- v2 = "C++ 可读的纯 npy 化消费子集" + 内存工程（u16/f32/lazy），meta.json 是装饰（C++ 不读），schema 靠自觉 → v3 必须机制化
- pickle 四宗罪：代码即 schema / 无版本 / 反序列化=RCE / 隐式契约
- 桥建在 python 侧（pickle 不对称性）：v1→v2 已有，v2→v1 待建
- SQLite 胜出的关键：已在依赖栈（rosbag2 默认 sqlite3），事务/单文件/全语言/ blob 内压缩
