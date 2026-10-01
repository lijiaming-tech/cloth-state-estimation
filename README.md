# cloth-state-estimation

UniClothDiff 复现与面向真机部署的布类状态估计。

本仓库基于论文作者公开的 [UniClothDiff](https://github.com/Tongxuan259/UniClothDiff)（CoRL 2025）
代码，针对 **VR-Folding 数据集**做了数据准备、模板构建、训练脚本与评估流程的补充，
目标是从**部分观测点云**恢复完整的衣物三维顶点状态。

> 上游原始 README 保留在 [`README_upstream.md`](README_upstream.md)，内容未改动。

---

## 1. 项目目标

给定一帧的四视角观察点云 `pcd`（(10000,3)）与一个同拓扑的规范模板 `q_temp`（(4434,3)），
用扩散模型恢复该帧完整的布料顶点 `q`（(4434,3)）。

模板顶点与目标顶点**逐一对应**：顶点 `i` 的预测对应顶点 `i` 的真值。
推理必须使用与训练一致的输入构造（详见 [`docs/data.md`](docs/data.md) 的"推理接口"一节），
否则形状不匹配。

## 2. 代码来源与改动

| 部分 | 来源 |
| --- | --- |
| `uniclothdiff/`、`train.py`、`args.py`、`configs/`、`third_party/diffusers` | 上游 `Tongxuan259/UniClothDiff`（commit `6d96533`），未改动 |
| `scripts/tmpl_patchify.py` | 上游 |
| `scripts/build_vr_tshirt_template.py` | 本仓库新增（自 VR-Folding NOCS 构建 4434 顶点模板） |
| `scripts/zarr_to_state_est_hdf5.py` | 本仓库新增（Zarr → 训练用 HDF5） |
| `scripts/train_vrfolding_from_scratch.py` | 本仓库新增（状态估计训练主脚本） |
| `scripts/train_vrfolding_state_est.py` | 本仓库新增（早期 2000 步版本，保留备查） |
| `scripts/resume_vrfolding.py` | 续训入口，必须显式指定 `--resume` |
| `scripts/vrfolding_protocol.py` | 训练与离线评估共用的多种子协议 |
| `scripts/resume_download.py` | 本仓库新增（带重试的完整数据集下载脚本） |
| `experiments/` | 本仓库新增（可核实的指标与运行记录） |

上游 `origin` 已重命名为 `upstream`，**不向作者仓库推送**。

## 3. 当前进度（截至 2026-09-26）

- ✅ 从 HuggingFace 取到 VR-Folding T-shirt 示例数据（35 帧，单条序列）
- ✅ 用 NOCS 重建 4434 顶点、100 patch 的模板
- ✅ 完成 23 帧 → HDF5 转换，验证数据加载器（21 训练 / 2 留出）
- ✅ 完成 12500 步扩散训练（计划 20000 步，**中途中断**）
- ✅ 完成 5 种子评估

**结果（同一条序列，非泛化）**：5 种子平均逐顶点误差约 `0.019 ~ 0.023`，
衣物本身约 `0.7` 单位，即误差约 **3%**；比"直接输出训练集平均姿态"的基线好 **3.3 ~ 4.2 倍**。

详细数据见 [`experiments/metrics.csv`](experiments/metrics.csv)，
限制与下一步见 [`docs/progress.md`](docs/progress.md)。

> ⚠️ **所有验证均来自同一条轨迹（`00068_Tshirt_000000`）。跨轨迹、跨衣物、
> 真实相机泛化尚未验证，不能据此宣称可用于真机。**

## 4. 环境

已验证环境：conda env `clothdiff`（Python 3.9.19，PyTorch 2.1.2 + CUDA 12.1，RTX 4090）。

`torkit3d` 的自定义 CUDA 算子（`sample_farthest_points`、`batch_index_select`）是本项目的**硬依赖**，
必须在目标机器上编译可用。完整安装步骤见 [`docs/setup.md`](docs/setup.md)。

## 5. 数据准备

需要自行下载 VR-Folding 示例数据（**本仓库不含数据集**）。流程：

1. 从 HuggingFace 下载 T-shirt 示例 Zarr → `docs/data.md`
2. 导出 NOCS → 构建模板：`python scripts/build_vr_tshirt_template.py`
3. Zarr → HDF5：`python scripts/zarr_to_state_est_hdf5.py`

数据字段含义、坐标系、模板对应关系见 [`docs/data.md`](docs/data.md)。

## 6. 训练

```bash
conda run --no-capture-output -n clothdiff \
  python -u scripts/train_vrfolding_from_scratch.py
```

源码根目录现在从脚本位置确定，数据默认位于仓库同级的
`vr_folding_state_est_23frames/`；可通过 `--data-dir` 指定。
每次运行写入新目录，保留历史权重。原始2026-09-26脚本可从commit `9208740` 查阅。

训练配置来自 `configs/train_state_est.yaml`，运行时覆盖：
micro-batch 2 × 累积 4（有效批上限8，末尾批次按实际样本数加权）、lr `1e-5`、1000 步线性 warmup + cosine 衰减到 0、
共 20000 步（实际中断于约 12700 步）。

默认五种子DDPM50评估，最佳权重保存为 `best_train_multiseed.pt`，以两帧训练的
平均归一化几何误差选择；这仍是拟合检查，不代表独立验证集最优。
续训必须显式指定 `--resume /absolute/path/to/latest.pt`，从检查点读取原配置，
可以用 `--max-updates 1000` 限制本轮预算而保持原学习率调度。
完整审查、改动和验证边界见 [`docs/training_review_2026-10-01.md`](docs/training_review_2026-10-01.md)。

## 7. 评估

```bash
conda run --no-capture-output -n clothdiff python experiments/eval_5seed.py \
  --run /home/agilex/ljm/dyf/vr_folding_capacity_runs/20260926_173508
```

历史run默认评估 `best_train.pt` 和 `latest.pt`；新run请显式指定
`--checkpoints best_train_multiseed.pt latest.pt`。结果写入run中的独立评估目录，
不会覆盖 `experiments/` 中的历史CSV。两个相邻留出帧均参与报告，均不参与选权重。

指标定义：
- `vertex_l2`：预测与真值的逐顶点欧氏距离均值（数据集原始单位）
- `vertex_l2_over_gt_bbox_diagonal`：上者除以真值包围盒对角线，用于跨帧比较
- `mean_train_pose_vertex_l2`：对照基线——直接输出训练集平均姿态的误差
- `chamfer_mean_unsquared`：对称 Chamfer 均值（不保留逐顶点对应）

## 8. 权重与数据的位置

**均不在本仓库中**，需单独备份，见 [`experiments/checkpoints.md`](experiments/checkpoints.md)。

## 9. 许可

沿用上游 MIT License，见 [`LICENSE`](LICENSE)。

## 10. 引用

```bibtex
@article{tian2025uniclothdiff,
  author    = {Tian, Tongxuan and Li, Haoyang and Ai, Bo and Yuan, Xiaodi and Huang, Zhiao and Su, Hao},
  title     = {Diffusion Dynamics Models with Generative State Estimation for Cloth Manipulation},
  journal   = {Conference on Robot Learning (CoRL)},
  year      = {2025},
}
```
