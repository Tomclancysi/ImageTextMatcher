"""检查 PyTorch CUDA 支持状态的脚本"""
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
    try:
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
    except:
        print("cuDNN version: N/A")
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

