import torch
from einops import repeat
from models.components.dataclass_config import NormlizationConfig, ParamInitConfig, PosEncodingConfig
from models.components.encoder_layers import TransformerEncoderModule
from models.components.encoder_modules import InputEncoder
from torch import nn

x = torch.randn(16, 2048, 1)
gridx = torch.linspace(0, 1, 2048)
gridx = gridx.reshape(1, 2048, 1)
pos = gridx.repeat([x.shape[0], 1, 2])
lens = torch.tensor([3, 5])
max_len = 8
mask = torch.arange(max_len).expand(len(lens), max_len) < lens.unsqueeze(1)
mask = mask.reshape([16, 1])
mask = repeat(mask, "b d -> b n d", n=2048)

scale = [1, 2, 3, 4, 5, 6, 7, 8]

model = InputEncoder(
    in_features=1,
    out_features=196,
    attn_type="standard",
    d_model=96,
    nhead=8,
    num_layers=8,
    dim_feedforward=96 * 2,
    dropout=0.1,
    activation="geglu",
    norm_first=True,
    bias=False,
    scale=scale,
    norm_config=NormlizationConfig(method="InstanceNorm"),
    init_config=ParamInitConfig(method="orthogonal", gain=1.0, diagonal_weight=1.0),
    pos_config=PosEncodingConfig(method="cat", dim=2),
)

out = model(x, pos, mask)
