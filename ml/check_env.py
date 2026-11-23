import sys
print("Python:", sys.version)

try:
    import torch
    print("torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("No GPU. CPU-only mode.")
except Exception as e:
    print("torch import failed:", e)

try:
    import timm
    print("timm:", timm.__version__)
except Exception as e:
    print("timm import failed:", e)

try:
    import segmentation_models_pytorch as smp
    print("smp:", smp.__version__)
except Exception as e:
    print("smp import failed:", e)

try:
    import pandas as pd
    print("pandas:", pd.__version__)
except Exception as e:
    print("pandas import failed:", e)

try:
    import matplotlib
    print("matplotlib:", matplotlib.__version__)
except Exception as e:
    print("matplotlib import failed:", e)

try:
    import numpy as np
    print("numpy:", np.__version__)
except Exception as e:
    print("numpy import failed:", e)

print("ALL CHECKS DONE")
