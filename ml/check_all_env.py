import torch, sys
try:
    import segmentation_models_pytorch as smp
    print('smp version:', smp.__version__)
except Exception as e:
    print('smp import error:', e)
try:
    import timm
    print('timm version:', timm.__version__)
except Exception as e:
    print('timm import error:', e)
try:
    import efficientnet_pytorch
    print('efficientnet-pytorch version:', efficientnet_pytorch.__version__)
except Exception as e:
    print('efficientnet-pytorch import error:', e)
print('torch version:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
