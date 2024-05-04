from torch import nn

from collections import OrderedDict

model = nn.Sequential(
    OrderedDict(
        [("conv1", nn.Conv2d(1, 20, 5)), ("relu1", nn.ReLU()), ("conv2", nn.Conv2d(20, 64, 5)), ("relu2", nn.ReLU())]
    )
)

for layer in model:
    print(layer)
