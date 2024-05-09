from typing import List, Literal, Optional, Union

from torch import Tensor, nn

from ..activation import GaussianFourierFeatureTransform
from ..dataclass_config import NormlizationConfig, ParamInitConfig, PosEncodingConfig
from .layers import CrossLinearTransformerEncoderModule, TransformerEncoderModule


__all__ = ["InputEncoder", "QueryEncoder", "CrossAttentionEncoder"]


class InputEncoder(nn.Module):
    def __init__(
        self,
        attn_type: Literal["fourier", "galerkin", "standard"],
        in_features: int,
        out_features: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        scale: Union[int, List[int]],
        dropout: float = 0.05,
        activation: Literal["relu", "gelu", "geglu"] = "relu",
        norm_first: bool = True,
        bias: bool = False,
        norm_config: NormlizationConfig = NormlizationConfig(method="InstanceNorm"),
        init_config: ParamInitConfig = ParamInitConfig(method="xavier", gain=1.0),
        pos_config: PosEncodingConfig = PosEncodingConfig(method="RoPE", dim=1),
    ) -> None:
        super(InputEncoder, self).__init__()

        self.norm_first: bool = norm_first

        self.in_embeddings = nn.Linear(in_features, d_model, bias=bias)

        self.dropout = nn.Dropout(p=dropout)

        self.encoder = TransformerEncoderModule(
            attn_type=attn_type,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            scale=scale,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias=bias,
            norm_config=norm_config,
            init_config=init_config,
            pos_config=pos_config,
        )

        self.out_embeddings = nn.Linear(d_model, out_features, bias=bias)

    def forward(self, x: Tensor, input_pos: Optional[Tensor] = None, mask: Optional[Tensor] = None) -> Tensor:
        ax_embedding: Tensor = self.in_embeddings(x)
        z: Tensor = self.encoder(ax_embedding, input_pos, mask)
        out_embeddng: Tensor = self.out_embeddings(z)

        return out_embeddng


class QueryEncoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        d_model: int,
        bias: bool = False,
        scale: int = 8,
    ) -> None:
        super(QueryEncoder, self).__init__()
        self.proj = nn.Sequential(
            GaussianFourierFeatureTransform(num_input_channels=in_features, mapping_size=d_model, scale=scale),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model, bias=bias),
        )

    def forward(self, query_pos: Tensor) -> Tensor:
        return self.proj(query_pos)


class CrossAttentionEncoder(nn.Module):
    def __init__(
        self,
        attn_type: Literal["fourier", "galerkin"],
        d_model: int,
        nhead: int,
        dim_feedforward: Optional[int] = None,
        num_layers: int = 1,
        scale: Union[int, List[int]] = 1,
        dropout: float = 0.1,
        activation: Literal["relu", "gelu", "geglu"] = "relu",
        norm_first: bool = True,
        bias: bool = False,
        residual: bool = True,
        norm_config: NormlizationConfig = NormlizationConfig(method="InstanceNorm"),
        init_config: ParamInitConfig = ParamInitConfig(method="xavier", gain=1.0),
        pos_config: PosEncodingConfig = PosEncodingConfig(method="RoPE", dim=1),
    ) -> None:
        super(CrossAttentionEncoder, self).__init__()

        self.encoder = CrossLinearTransformerEncoderModule(
            attn_type=attn_type,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias=bias,
            scale=scale,
            residual=residual,
            norm_config=norm_config,
            init_config=init_config,
            pos_config=pos_config,
        )

    def forward(
        self,
        query_emb: Tensor,
        input_emb: Tensor,
        query_pos: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        return self.encoder(query_emb, input_emb, query_pos, input_pos, mask)
