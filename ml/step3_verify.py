"""STEP 3: Verification of CUDA PyTorch and GPU execution."""
import sys

print("=" * 60)
print("STEP 3: GPU Verification")
print("=" * 60)

try:
    import torch
    print(f"torch.__version__          : {torch.__version__}")
    print(f"torch.cuda.is_available()  : {torch.cuda.is_available()}")
    print(f"torch.version.cuda         : {getattr(torch.version, 'cuda', 'None')}")

    if not torch.cuda.is_available():
        print("\nERROR: torch.cuda.is_available() returned False!")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    print(f"GPU device 0               : {device_name}")
    print(f"Total VRAM                 : {total_mem:.0f} MiB")

    # Tiny CUDA tensor operation test
    print("\nRunning tiny CUDA tensor test...")
    x = torch.randn(1000, 1000, device="cuda")
    y = torch.matmul(x, x)
    torch.cuda.synchronize()
    print(f"CUDA tensor test PASSED! Result shape: {y.shape}, device: {y.device}")
    del x, y
    torch.cuda.empty_cache()

except Exception as e:
    print(f"\nERROR in torch verification: {e}")
    sys.exit(1)

print("\nVerifying model libraries...")
try:
    import segmentation_models_pytorch as smp
    print(f"segmentation_models_pytorch : {smp.__version__} (OK)")
except Exception as e:
    print(f"segmentation_models_pytorch : FAILED ({e})")
    sys.exit(1)

try:
    import timm
    print(f"timm                        : {timm.__version__} (OK)")
except Exception as e:
    print(f"timm                        : FAILED ({e})")
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL VERIFICATIONS PASSED!")
print("=" * 60)
