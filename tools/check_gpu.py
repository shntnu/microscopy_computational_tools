#!/usr/bin/env python
import sys
import platform

print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")

try:
    import torch

    print(f"PyTorch version: {torch.__version__}")

    # Check if CUDA is available (for non-Mac systems)
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")

    # Check if MPS (Metal Performance Shaders) is available (for macOS)
    if hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
        print(f"PyTorch MPS (Metal) available: {torch.mps.is_available()}")
    else:
        print("PyTorch MPS not supported in this PyTorch version")

    # Print the device that will be used
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif (
        hasattr(torch, "mps")
        and hasattr(torch.mps, "is_available")
        and torch.mps.is_available()
    ):
        device = torch.device("mps")
        print("Using MPS (Metal) device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    # Test with a simple operation
    x = torch.randn(1000, 1000)
    print(f"Tensor created on: {x.device}")

    # Move to detected device and perform operation
    x = x.to(device)
    print(f"Tensor moved to: {x.device}")

    # Perform a computation to test device
    start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

    if start is not None:
        start.record()

    # Matrix multiplication
    result = torch.matmul(x, x)

    if end is not None:
        end.record()
        torch.cuda.synchronize()
        print(f"Operation time: {start.elapsed_time(end)} ms")

    print("GPU computation successful")

except ImportError:
    print("PyTorch not installed")
except Exception as e:
    print(f"Error: {e}")
