import torch
from torch import autograd
import rootutils

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.models.components import OperatorFormer, SimpleOperatorLearningL2Loss

torch.autograd.set_detect_anomaly(True)


net = OperatorFormer()
loss_fn = SimpleOperatorLearningL2Loss()
x = torch.randn(16, 2048, 1)
input_pos = torch.randn(16, 2048, 1)
query_pos = torch.randn(16, 2048, 1)

y = torch.randn(16, 2048, 1)

z = net(x, input_pos, query_pos)

loss = loss_fn(z, y)

with torch.autograd.detect_anomaly():
    loss.backward()
