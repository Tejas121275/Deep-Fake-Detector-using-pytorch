import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
print(f"cuDNN Version: {torch.backends.cudnn.version()}")