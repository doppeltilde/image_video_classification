import torch
import pytorch_ocl

torch.randn(10, 10, device="ocl:0")
