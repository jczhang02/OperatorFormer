import torch
from einops import repeat
from models.components.attention import CrossLinearMultiHeadAttention
from models.components.dataclass_config import NormlizationConfig, ParamInitConfig, PosEncodingConfig
from models.components.encoder import CrossAttentionEncoder, InputEncoder
from models.components.encoder.layers import (
    CrossLinearTransformerEncoderLayer,
    CrossLinearTransformerEncoderModule,
    TransformerEncoderModule,
)
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

model = CrossLinearMultiHeadAttention(
    attn_type="fourier",
    d_model=96,
    nhead=8,
    dropout=0.1,
    bias=False,
    norm_config=NormlizationConfig(method="InstanceNorm"),
    init_config=ParamInitConfig(method="orthogonal", gain=1.0, diagonal_weight=1.0),
    pos_config=PosEncodingConfig(method="cat", dim=2),
)

query_emb = torch.randn(16, 2048, 96)
input_emb = torch.randn(16, 2048, 96)
query_pos = pos
input_pos = pos

# out = model(query_emb, input_emb, query_pos, input_pos, mask)


model2 = CrossLinearTransformerEncoderLayer(
    attn_type="fourier",
    d_model=96,
    nhead=8,
    dim_feedforward=96,
    dropout=0.1,
    bias=False,
    residual=False,
    norm_config=NormlizationConfig(method="InstanceNorm"),
    init_config=ParamInitConfig(method="orthogonal", gain=1.0, diagonal_weight=1.0),
    pos_config=PosEncodingConfig(method="cat", dim=2),
)

model3 = CrossLinearTransformerEncoderModule(
    attn_type="fourier",
    d_model=96,
    nhead=8,
    dim_feedforward=96,
    num_layers=8,
    dropout=0.1,
    bias=False,
    scale=scale,
    residual=True,
    norm_config=NormlizationConfig(method="InstanceNorm"),
    init_config=ParamInitConfig(method="orthogonal", gain=1.0, diagonal_weight=1.0),
    pos_config=PosEncodingConfig(method="cat", dim=2),
)

model4 = CrossAttentionEncoder(
    attn_type="fourier",
    d_model=96,
    nhead=8,
    dim_feedforward=96,
    num_layers=1,
    dropout=0.1,
    activation="relu",
    norm_first=True,
    bias=False,
    scale=1,
    residual=True,
    norm_config=NormlizationConfig(method="InstanceNorm"),
    init_config=ParamInitConfig(method="orthogonal", gain=1.0, diagonal_weight=1.0),
    pos_config=PosEncodingConfig(method="cat", dim=2),
)

out = model4(query_emb, input_emb, query_pos, input_pos, mask)
print(out.shape)
