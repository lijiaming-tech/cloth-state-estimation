# 权重与数据位置

**本仓库不含权重和数据集。** 以下文件只存在于本机，需要单独备份。

所有校验值均由本机实测（`sha256sum`）。**没有任何异地备份。**

## 1. 模型权重

运行目录：`/home/agilex/ljm/dyf/vr_folding_capacity_runs/20260926_173508/`

| 文件 | 内部 step | 大小 | SHA-256 |
| --- | ---: | ---: | --- |
| `latest.pt` | **12500** | 2,664,257,878 B（约 2.5 GB） | `2eb2736b1b840b502e75acfd4558338e96c608dde7a3eda899e9a777d8b46f51` |
| `best_train.pt` | **12000** | 888,086,192 B（约 846 MB） | `7503a86ec43bedb2958adc131e18904ff2fdcb529cbba35b59bdc08525ad1848` |

说明：

- `latest.pt` 含**模型 + 优化器状态**，因此体积大；
  `best_train.pt` 只含**模型权重**。
- 两者内部记录的实际 step 已用 `torch.load(...)['step']` 核实，与文件名无关。
- `best_train.pt` 的选择标准是训练时的**单次采样**评估（噪声较大），
  5 种子复评显示 step 12500 实际略优于 step 12000（配对差值 4/5 种子支持，t≈−2.32, p≈0.08，
  **未达统计显著**）。`latest.pt` 可用于续训和下一轮对照，尚无生产/真机使用证据。

### 训练配置对应关系

两个权重都由 `experiments/run_20260926_173508_config.json` 描述的运行产生：

- 数据：`vr_folding_state_est_23frames`，21 训练帧（`000045 … 000145`）
- micro-batch 2 × 累积 4 = 有效批 8
- lr `1e-5`，1000 步线性 warmup 后 cosine 衰减到 0
- 计划 20000 步，**实际中断于约 12700 步**（`latest.pt` 停在 12500）
- 上游 commit：`6d965335cb43fbea1982eb5f441582dac3381fb2`

## 2. 模板

| 文件 | 大小 | SHA-256 |
| --- | ---: | --- |
| `assets/vr_tshirt_voronoi_template.pkl` | 151,286 B | `01267e287fc9d3d87082cfb2cb847df4e3eb71f86d3266331c127f0dc57bfced` |
| `assets/Tshirt.obj` | 673,826 B | `42e964b325d8306d8e607405b51a3a44960a9c0a4159169b1ca9a10c13653f8d` |

模板已在仓库内（体积小）。**权重与模板必须配套使用**：换模板需重新训练。

## 3. 数据集（仓库外）

| 内容 | 路径 | 大小 |
| --- | --- | ---: |
| 训练用 HDF5（23 帧） | `/home/agilex/ljm/dyf/vr_folding_state_est_23frames/` | 9.2 MB |
| 原始 Zarr 示例（35 帧） | `/home/agilex/ljm/dyf/uniClothDiff/data/data_examples/VR_Folding/` | 35 MB |
| **完整 folding 数据集** | `/home/agilex/ljm/dyf/vr_folding_download/data/folding/` | **128.346 GiB（269 个文件）** |
| flattening 数据集 | 未下载 | 约 360 GB |

**完整 folding 数据集（2026-10-01 下载完成，已全量校验）**：

- 269 个文件，`137,810,105,347` 字节，**sha256 全部匹配 HF LFS 官方哈希**
- 分片 `folding_dataset.z01`~`z268` 编号无缺口、无重复
- **尚未解压**；解压后需先确认内部 episode 结构，见 `docs/progress.md` 第 3 节

小数据集（前两行）可由 `docs/data.md` 中的步骤重新下载生成，**不必备份**。
完整 folding 数据集同样可重新下载，但耗时约 1 小时 40 分钟且依赖可用代理，
是否单独备份可视磁盘情况决定。

## 4. 运行记录（已入库，体积小）

| 文件 | 说明 |
| --- | --- |
| `experiments/run_20260926_173508_config.json` | 该次运行的完整配置 |
| `experiments/run_20260926_173508_metrics.jsonl` | 训练过程中每 1000 步的单次采样评估 |
| `experiments/run_20260926_173508_loss.csv` | 每步训练噪声 MSE 与学习率 |
| `experiments/eval_5seed_results.csv` | 5 种子复评的逐种子原始结果 |
| `experiments/metrics.csv` | 上表的汇总 |

未入库：`*.npz`（每帧每次采样的预测顶点数组，约 150 KB/个，共 30 个）。
如需复现逐顶点对比，可用 `experiments/eval_5seed.py` 重新生成。

## 5. 建议的备份优先级

1. **`latest.pt`**（2.5 GB）——含优化器的可续训权重，不能仅从代码恢复同一训练产物
2. `best_train.pt`（846 MB）——可选，与上者性能接近
3. 数据集——可重新下载，优先级最低

> ⚠️ `latest.pt` 与 `best_train.pt` **目前只有本机一份**，
> 且所在目录不属任何备份机制。建议尽快复制到独立位置。
