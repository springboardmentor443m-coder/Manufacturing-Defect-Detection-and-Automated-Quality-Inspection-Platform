"""STEP 1 – Diagnose the current Python/ML environment."""
import sys, importlib

print("=" * 60)
print("STEP 1: Environment Diagnosis")
print("=" * 60)
print(f"Python version : {sys.version}")
print(f"Python path    : {sys.executable}")
print()

# torch
try:
    import torch
    print(f"torch installed : YES")
    print(f"torch.__version__       : {torch.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.version.cuda      : {getattr(torch.version, 'cuda', 'None')}")
    is_cpu = "+cpu" in torch.__version__ or not torch.cuda.is_available()
    print(f"CPU-only build?         : {'+cpu' in torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU name                : {torch.cuda.get_device_name(0)}")
        total = torch.cuda.get_device_properties(0).total_mem / 1024**2
        print(f"Total VRAM              : {total:.0f} MiB")
except ImportError:
    print("torch installed : NO")
except Exception as e:
    print(f"torch error: {e}")

print()

# torchvision
for pkg_name, import_name in [
    ("torchvision", "torchvision"),
    ("segmentation-models-pytorch", "segmentation_models_pytorch"),
    ("timm", "timm"),
    ("numpy", "numpy"),
    ("opencv-python", "cv2"),
    ("Pillow", "PIL"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
]:
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", "unknown")
        print(f"{pkg_name:30s} : {ver}")
    except ImportError:
        print(f"{pkg_name:30s} : NOT INSTALLED")
    except Exception as e:
        print(f"{pkg_name:30s} : ERROR - {e}")

print()
print("=" * 60)
print("Diagnosis complete.")
print("=" * 60)
