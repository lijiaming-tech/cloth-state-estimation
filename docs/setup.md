# 环境安装

本文记录**实际验证过**的环境。版本号来自本机 `clothdiff` 环境实测，不是照抄上游声明。

## 已验证环境

| 组件 | 版本 |
| --- | --- |
| 操作系统 | Ubuntu（内核 5.15.0-102-generic） |
| GPU | NVIDIA GeForce RTX 4090，24 GB |
| conda 环境名 | `clothdiff` |
| Python | 3.9.19 |
| PyTorch | 2.1.2 |
| CUDA | 12.1（`torch.version.cuda`），`torch.cuda.is_available() == True` |
| diffusers | 0.29.0.dev0（来自本仓库 `third_party/diffusers`，非 PyPI 版本） |
| numpy | 1.26.4 |
| h5py | 3.11.0 |
| open3d | 0.19.0 |
| trimesh | 4.12.2 |
| scipy | 1.13.1 |
| zarr | 2.18.2 |
| numcodecs | 0.12.1 |

## 安装步骤

### 1. 创建 conda 环境

仓库内含上游提供的 `environment.yml`（固定了完整依赖版本）：

```bash
conda env create -f environment.yml
conda activate clothdiff
```

### 2. 安装 torkit3d（关键，必须编译 CUDA 扩展）

```bash
pip install "git+https://github.com/Jiayuan-Gu/torkit3d.git"
```

**本项目依赖 torkit3d 的自定义 CUDA 算子**，尤其是：

- `torkit3d.ops.sample_farthest_points.sample_farthest_points`（FPS）
- `torkit3d.nn.functional.batch_index_select`

安装后必须验证扩展确实可用，否则模板构建和训练都会失败：

```bash
python -c "
import torch
from torkit3d.ops.sample_farthest_points import sample_farthest_points
x = torch.rand(1, 1000, 3).cuda()
print(sample_farthest_points(x, 100).shape)   # 期望 torch.Size([1, 100])
"
```

> 若报 `ImportError: ... sample_farthest_points_cuda`，说明 CUDA 扩展未编译成功，
> 需要在目标机器上重新编译 torkit3d，而不是重装 PyTorch。

### 3. 安装本仓库自带的 diffusers 分支

本仓库的模型代码依赖 `third_party/diffusers` 中的**定制分支**（普通 PyPI 的 diffusers 不含
`BasicTransformerBlock` 的相同接口）。安装：

```bash
cd third_party/diffusers
pip install .
cd ../..
```

验证：

```bash
python -c "import diffusers; print(diffusers.__version__)"   # 期望 0.29.0.dev0
```

### 4. 安装本仓库

```bash
pip install -e .
```

### 5. 读取 Zarr 数据所需的额外依赖

`zarr` 与 `numcodecs` 不在上游 `environment.yml` 中，但读取 VR-Folding 原始 Zarr 时需要：

```bash
pip install 'zarr>=2.16,<3' 'numcodecs<0.16'
```

> ⚠️ 本机的 `clothdiff` 环境是先装好的，`zarr` 是事后补装的。
> 若要**从零重建**环境，请把这两项加入 `environment.yml` 后再创建。
> 注意 `numcodecs` 需 `<0.16`，较新版本与 `zarr 2.x` 不兼容。

## 环境分工（本机实际用法）

本机存在两个环境，用途不同：

| 环境 | 用途 | 关键包 |
| --- | --- | --- |
| `clothdiff` | 模板构建、训练、评估 | PyTorch 2.1.2 + torkit3d CUDA 算子 |
| `ler` | 仅用于 Zarr → HDF5 转换 | zarr 2.18.3、h5py 3.16.0 |

原因：`torkit3d` 只在 `clothdiff` 里，而 `zarr` 一度只装在 `ler` 里，两者不通用。
脚本 `scripts/zarr_to_state_est_hdf5.py` 因此设计为可在任一有 zarr 的环境中运行，
其输出 HDF5 再由 `clothdiff` 环境读取。

## 已知问题

1. **`piper_sdk` 无关**——本项目不涉及机器人控制，无需该依赖。
2. **网络**：本机访问 PyPI 走镜像时常出现 SSL 错误，实测 `-i https://pypi.org/simple` 可用。
3. **`zarr` 版本**：必须 `<3`。`zarr 3.x` 的 API 与本研究使用的 2.x 不兼容。
