# PyTorch CUDA 支持设置指南

## 问题说明

即使安装了 `nvidia-cublas-cu12`、`nvidia-cuda-runtime-cu12`、`nvidia-cudnn-cu12` 这些包，PyTorch 仍然无法使用 CUDA，因为：

1. **PyTorch 需要编译时支持 CUDA**：PyTorch 本身必须是支持 CUDA 的版本
2. **CUDA 驱动版本匹配**：系统的 NVIDIA 驱动版本需要与 PyTorch 要求的 CUDA 版本兼容

## 解决方案

### 步骤 1: 检查 NVIDIA 驱动和 CUDA 版本

```bash
# 检查 NVIDIA 驱动版本
nvidia-smi

# 检查系统 CUDA 版本（如果有安装）
nvcc --version
```

### 步骤 2: 卸载当前的 PyTorch 和 torchvision

```bash
pip uninstall torch torchvision torchaudio
```

### 步骤 3: 安装支持 CUDA 的 PyTorch

根据你的 CUDA 版本，从 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取安装命令。

**对于 CUDA 12.x（推荐）：**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**对于 CUDA 11.8：**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**如果系统没有 NVIDIA GPU 或驱动：**
```bash
# 安装 CPU 版本（当前状态）
pip install torch torchvision torchaudio
```

### 步骤 4: 验证安装

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
```

## 注意事项

1. **CPU 版本 vs CUDA 版本**：
   - CPU 版本：`pip install torch`（从 PyPI 安装）
   - CUDA 版本：`pip install torch --index-url https://download.pytorch.org/whl/cu121`（从 PyTorch 官方源安装）

2. **CUDA 运行时库**：
   - `nvidia-cublas-cu12` 等包是 CUDA 运行时库，它们**不能**让 CPU 版本的 PyTorch 使用 CUDA
   - 这些包只在安装 CUDA 版本的 PyTorch 时作为依赖自动安装

3. **Windows 系统**：
   - 确保已安装 NVIDIA 驱动
   - 确保驱动版本支持所需的 CUDA 版本

## 快速检查脚本

运行以下 Python 脚本检查当前状态：

```python
import torch
import sys

print("=" * 50)
print("PyTorch CUDA 支持检查")
print("=" * 50)
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("\n⚠️  PyTorch 无法使用 CUDA")
    print("可能的原因：")
    print("1. 安装的是 CPU 版本的 PyTorch")
    print("2. 系统没有 NVIDIA GPU 或驱动未安装")
    print("3. CUDA 驱动版本不匹配")
    print("\n解决方案：")
    print("1. 卸载当前 PyTorch: pip uninstall torch torchvision torchaudio")
    print("2. 安装 CUDA 版本: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
print("=" * 50)
```

