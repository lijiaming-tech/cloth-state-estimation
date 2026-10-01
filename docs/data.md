# 数据说明

## 1. 数据来源

| 项 | 值 |
| --- | --- |
| HuggingFace 数据集 | [`robotflow/vr-folding`](https://huggingface.co/datasets/robotflow/vr-folding)（`repo_type=dataset`） |
| 使用的 revision | `2d49d4ca4c9f7c6d8f97daedbeea99ccbb024878` |
| 示例子树 | `data/data_examples/VR_Folding/vr_simulation_folding_dataset_example.zarr/Tshirt` |
| 官方大集 | `data/folding`（269 卷 ≈ **137 GB**）、`data/flattening`（707 卷 ≈ **360 GB**） |

**完整数据集是分卷 zip（`.z01`/`.z02`/…），必须下齐所有卷才能解出任何内容，无法只挑几个
episode。** 本仓库所有实验只用了示例子树。

### 下载示例子树（只取需要的帧，避免全量）

```bash
python -c '
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="robotflow/vr-folding", repo_type="dataset",
    revision="2d49d4ca4c9f7c6d8f97daedbeea99ccbb024878",
    allow_patterns="data/data_examples/VR_Folding/vr_simulation_folding_dataset_example.zarr/Tshirt/samples/00068_Tshirt_000000_*/**",
    local_dir="/path/to/data_root", max_workers=8)
'
```

> 实测：`max_workers=1` 会卡死（5 分钟无进展），必须多线程。整个示例约 30 MB。

### 下载完整 folding 数据集

`scripts/resume_download.py` 是一个带指数退避重试的下载脚本，用于拉取完整的
`data/folding/*`（约 137 GB）。因为分卷 zip 必须下齐才能解压，这个脚本会长时间运行。

**本机状态（2026-10-01 核实）**：该脚本已连续运行约 **4 天 18 小时**，
下载到约 **37 GB**，且**当前卡在重试循环中**（日志显示 `attempt 1276`，
反复出现 `SSLZeroReturnError ... huggingface.co`，每次退避 300 秒）。

> 该 SSL 错误与本机代理有关：实测环境变量中的
> `http_proxy` / `https_proxy`（`http://127.0.0.1:7897`）会让经它的 HTTPS 请求全部失败，
> 而**绕过代理直连**时 `api.github.com` 等可正常访问。
> 若下载长期无进展，应检查代理是否可用，或为该进程显式指定可用的代理。
> 注意 HuggingFace 在部分网络环境下**需要**代理，不能简单地去代理解决。

## 2. 示例数据的实际内容（已核实）

**示例子树只有 35 个 sample，且全部来自同一条序列 `00068_Tshirt_000000`**，
帧号 `000045 → 000215`，步长 5。

```
Tshirt/samples/00068_Tshirt_000000_000045/
    mesh/cloth_verts               (4434, 3)  float32
    mesh/cloth_nocs_verts          (4434, 3)  float32
    mesh/cloth_faces_tri           (8312, 3)  int32
    point_cloud/point              (30000, 3) float16   ← 注意是 float16
    point_cloud/rgb                (30000, 3) uint8
    point_cloud/nocs               (30000, 3) float16
    point_cloud/sizes              (4,)       int64     = [7500, 7500, 7500, 7500]
    hand_pose/, grip_vertex_id/, marching_cube_mesh/
```

| 字段 | 含义 |
| --- | --- |
| `mesh/cloth_verts` | 该帧完整布料顶点，加载后即 `q_gt` |
| `mesh/cloth_nocs_verts` | 规范化物体坐标（NOCS）。**实测帧间完全一致，逐点最大差 `0.000e+00`** |
| `mesh/cloth_faces_tri` | 三角面片索引，范围 `0 ~ 4433` |
| `point_cloud/point` | 四视角拼接的观察点云，`sizes` 表示每个视角 7500 点 |
| `point_cloud/sizes` | **四个视角的点数，不是四个时序帧** |

> ⚠️ `point_cloud/point` 在 Zarr 中以 **float16** 存储。转换时必须显式转 float32；
> float16 只有约 3 位十进制有效数字，直接用作条件点云会引入精度损失。

**示例数据里不存在跨序列样本。** `instances/` 下有 `00068` 和 `00074` 两个实例目录，
但 `samples/` 只有 `00068` 的帧。**因此用这份数据无法做任何跨 episode 泛化验证。**

## 3. 转换后的训练用 HDF5

由 `scripts/zarr_to_state_est_hdf5.py` 生成，每个 sample 一个文件，文件名为 sample 名
（因此按文件名排序即为时间顺序）：

| 键 | 形状 | 来源 | 说明 |
| --- | --- | --- | --- |
| `q` | (4434, 3) float32 | `mesh/cloth_verts` | 扩散模型的训练目标 |
| `points` | (30000, 3) float32 | `point_cloud/point` | 由 float16 加宽 |
| `faces` | (8312, 3) int32 | `mesh/cloth_faces_tri` | 数据加载器不使用，保留以便导出带面片的网格 |

数据加载器 `ClothStateEstDataset` 只读 `points` 与 `q`：
加载时对 `points` 做 jitter（`points_jitter_sigma`）后随机采样 10000 个得到 `pcd`。

### 数据划分

`ClothStateEstDataset` 按**文件名字排序**，取前 95% 为训练、末尾 5% 为验证：

| 帧数 | 训练 | 留出 |
| --- | --- | --- |
| 23（本仓库实际使用） | 21（`000045 … 000145`） | 2（`000150`、`000155`） |
| 35（完整示例） | 33 | 2（`000210`、`000215`） |

> ⚠️ **留出帧与训练帧同属一条序列且时序相邻。**
> 实测 `000145` 与 `000150` 的真值逐顶点平均距离仅 **0.0009**（衣物几乎静止），
> 因此"验证集"指标**不能当作泛化证据**，只能视为同序列的重复测量。

## 4. 模板（`q_temp`）

### 构建方法

`scripts/build_vr_tshirt_template.py`，从 NOCS 顶点直接构建：

1. `points = nocs * 0.5`
2. `points -= points.mean(axis=0)`（居中）
3. 最远点采样取 100 个中心
4. 每个顶点分配到最近中心（Voronoi），得到 100 个 patch

产物字段（与上游 `scripts/tmpl_patchify.py` 完全一致）：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `points` | (4434, 3) | 归一化后的模板顶点，**即 `q_temp`** |
| `centers` | (100, 3) | 100 个 patch 中心 |
| `center_idx` | (100,) | 中心在 `points` 中的索引 |
| `patch_index` | list[100] | 每个 patch 包含的顶点索引 |
| `patch_points` | list[100] | 每个 patch 的顶点坐标 |

**已验证**：4434 个顶点被 100 个 patch 完整覆盖且无重复；patch 大小 18 ~ 95，均值 44.3。

### 关键：顶点顺序必须一致

模板顶点 `i` 与 `q_gt` 顶点 `i` **一一对应**。因此：

- **不要**用 `assets/Tshirt.obj` 经 `trimesh.load` 重新生成模板——
  实测 `trimesh` 会把 4434 个顶点合并成 4252 个，导致索引错位。
- 修改模板的 patch 划分意味着**必须重新训练**，旧权重不再兼容。

### 尺度关系（实测）

| | 各轴跨度 (x, y, z) |
| --- | --- |
| 模板 `q_temp` | 0.157, 0.098, 0.198 |
| 场景 `q_gt`（如帧 150） | 0.401, 0.078, 0.640 |

模板比场景小约 **3.5 ~ 4 倍**，且是**规范（未折叠）**形状，而场景中的衣物是折叠态。
轴序一致（z > x > y），是一个近似各向同性的缩放差。

## 5. 推理接口（重要，易错）

模型的输入构造在**训练路径**与**仓库自带 pipeline 的默认路径**之间不一致：

| 路径 | 输入构造 | 形状 |
| --- | --- | --- |
| `DDPM_StateEst.training_losses_with_cfg`（训练实际使用） | `torch.cat([noisy_input, q_temp], dim=-1)` | `(B, N, 6)` |
| `ClothStateEstPipeline._call_v2`（推理，需 `call_v2=True`） | 同上 | `(B, N, 6)` |
| `ClothStateEstPipeline.__call__`（默认路径） | `torch.cat([q_temp.unsqueeze(1), x.unsqueeze(1)], dim=1)` | `(B, 2, N, 3)` |

模型首层是 `nn.Linear(in_channels=6, ...)`，**默认路径的 4 维张量会导致形状错误**。
本仓库的训练与评估脚本都使用第一种构造，与训练一致。

## 6. 坐标系与单位

**未作完整严格核验。** 已知：

- `q_gt` 与 `pcd` 在同一坐标系下，数值范围接近（例如帧 150：`q` 范围
  `[-0.234, 0.569, -0.779] ~ [0.166, 0.647, -0.140]`）
- 所有指标以**数据集原始单位**报告，**不应声称是经过标定的米**
- 相机到机器人基座的转换、以及到真机部署坐标系的映射，尚未验证

## 7. 真机 RGB-D 数据

本仓库目前**不含**真机 RGB-D 数据或相关代码。VR-Folding 是仿真数据，
从仿真到真实相机的迁移尚未开始。
