from typing import List, Literal, Optional, Union

from torch import Tensor, nn

from ..attention import (
    CrossLinearMultiHeadAttention,
    FeedForward,
    LinearMultiHeadAttention,
    StandardMultiHeadAttention,
)
from ..dataclass_config import NormlizationConfig, ParamInitConfig, PosEncodingConfig
from ..functional import _get_clones


__all__ = [
    "TransformerEncoderModule",
    "StandardTransformerEncoderModule",
    "StandardTransformerEncoderLayer",
    "LinearTransformerEncoderModule",
    "LinearTransformerEncoderLayer",
    "CrossLinearTransformerEncoderModule",
    "CrossLinearTransformerEncoderLayer",
]


class TransformerEncoderModule(nn.Module):
    def __init__(
        self,
        attn_type: Literal["standard", "fourier", "galerkin"],
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        scale: Union[int, List[int]] = 1,
        dropout: float = 0.1,
        activation: Literal["relu", "gelu", "geglu"] = "relu",
        norm_first: bool = False,
        bias: bool = False,
        norm_config: NormlizationConfig = NormlizationConfig(method="InstanceNorm"),
        init_config: ParamInitConfig = ParamInitConfig(method="xavier", gain=1.0),
        pos_config: PosEncodingConfig = PosEncodingConfig(method="RoPE", dim=1),
    ) -> None:
        super(TransformerEncoderModule, self).__init__()

        self.attn_type = attn_type

        if attn_type == "standard":
            self.transformer = StandardTransformerEncoderModule(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                num_layers=num_layers,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
                bias=bias,
            )
        else:
            self.transformer = LinearTransformerEncoderModule(
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

    def forward(self, x: Tensor, pos: Optional[Tensor] = None, mask: Optional[Tensor] = None) -> Tensor:
        return self.transformer(x, pos, mask)


class StandardTransformerEncoderModule(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float = 0.1,
        activation: Literal["relu", "gelu", "geglu"] = "relu",
        norm_first: bool = True,
        bias: bool = False,
    ) -> None:
        super(StandardTransformerEncoderModule, self).__init__()

        encoder_layer = StandardTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias=bias,
        )

        self.layers = _get_clones(encoder_layer, num_layers=num_layers)

    def forward(self, x: Tensor, pos: Optional[Tensor] = None, mask: Optional[Tensor] = None) -> Tensor:
        output: Tensor = x
        for mod in self.layers:
            output = mod(output, pos, mask)
        return output


class StandardTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: Literal["relu", "gelu", "geglu"] = "relu",
        norm_first: bool = True,
        bias: bool = False,
    ) -> None:
        super(StandardTransformerEncoderLayer, self).__init__()

        self.norm_first: bool = norm_first

        self.norm1 = nn.LayerNorm(normalized_shape=d_model, bias=bias)

        self.self_attn = StandardMultiHeadAttention(d_model=d_model, nhead=nhead, dropout=dropout, bias=bias)
        self.dropout = nn.Dropout(p=dropout)

        self.norm2 = nn.LayerNorm(normalized_shape=d_model, bias=bias)
        self.feedforward = FeedForward(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            activation=activation,
            dropout=dropout,
            bias=bias,
        )

    def forward(self, src: Tensor, src_pos: Optional[Tensor] = None, mask: Optional[Tensor] = None) -> Tensor:
        if self.norm_first:
            output: Tensor = self.norm1(src)
            output = self.self_attn(output, src_pos, mask) + output
            output = self.norm2(output)
            output = self.feedforward(output) + output
        else:
            output: Tensor = self.self_attn(src, src_pos, mask) + src
            output = self.feedforward(output) + output

        return output


class LinearTransformerEncoderModule(nn.Module):
    def __init__(
        self,
        attn_type: Literal["fourier", "galerkin"],
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        scale: Union[int, List[int]] = 1,
        dropout: float = 0.1,
        activation: Literal["relu", "gelu", "geglu"] = "relu",
        norm_first: bool = True,
        bias: bool = False,
        norm_config: NormlizationConfig = NormlizationConfig(method="InstanceNorm"),
        init_config: ParamInitConfig = ParamInitConfig(method="xavier", gain=1.0),
        pos_config: PosEncodingConfig = PosEncodingConfig(method="RoPE", dim=1),
    ) -> None:
        super(LinearTransformerEncoderModule, self).__init__()

        if isinstance(scale, int):
            scale = [scale] * num_layers
        assert len(scale) == num_layers, f"The number of layers is supposed to be equal to {len(scale)}."

        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            setattr(pos_config, "scale", scale[layer_idx])

            layer = LinearTransformerEncoderLayer(
                attn_type=attn_type,
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
                bias=bias,
                norm_config=norm_config,
                init_config=init_config,
                pos_config=pos_config,
            )
            self.layers.append(layer)

    def forward(self, x: Tensor, pos: Optional[Tensor] = None, mask: Optional[Tensor] = None) -> Tensor:
        output: Tensor = x
        for mod in self.layers:
            output = mod(output, pos, mask)
        return output


class LinearTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        attn_type: Literal["fourier", "galerkin"],
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: Literal["relu", "gelu", "geglu"] = "relu",
        norm_first: bool = True,
        bias: bool = True,
        norm_config: NormlizationConfig = NormlizationConfig(method="InstanceNorm"),
        init_config: ParamInitConfig = ParamInitConfig(method="xavier", gain=1.0, diagonal_weight=1.0),
        pos_config: PosEncodingConfig = PosEncodingConfig(method="RoPE", dim=1, scale=1),
    ) -> None:
        super(LinearTransformerEncoderLayer, self).__init__()

        self.norm_first: bool = norm_first
        self.norm1 = nn.LayerNorm(normalized_shape=d_model, bias=bias)
        self.self_attn = LinearMultiHeadAttention(
            attn_type=attn_type,
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            bias=bias,
            norm_config=norm_config,
            init_config=init_config,
            pos_config=pos_config,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model, bias=bias)
        self.feedforward = FeedForward(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            activation=activation,
            dropout=dropout,
            bias=bias,
        )

    def forward(self, x: Tensor, pos: Optional[Tensor] = None, mask: Optional[Tensor] = None) -> Tensor:
        if self.norm_first:
            output: Tensor = self.norm1(x)
            output = self.self_attn(output, pos, mask) + output
            output = self.norm2(output)
            output = self.feedforward(output) + output
        else:
            output: Tensor = self.self_attn(x, pos, mask) + x
            output = self.feedforward(output) + output

        return output


class CrossLinearTransformerEncoderModule(nn.Module):
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
        super(CrossLinearTransformerEncoderModule, self).__init__()

        if isinstance(scale, int):
            scale = [scale] * num_layers
        assert len(scale) == num_layers, f"The number of layers is supposed to be equal to {len(scale)}."

        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            setattr(pos_config, "scale", scale[layer_idx])

            layer = CrossLinearTransformerEncoderLayer(
                attn_type=attn_type,
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
                bias=bias,
                residual=residual,
                norm_config=norm_config,
                init_config=init_config,
                pos_config=pos_config,
            )
            self.layers.append(layer)

    def forward(
        self,
        query_emb: Tensor,
        input_emb: Tensor,
        query_pos: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        output: Tensor = input_emb
        for mod in self.layers:
            # TODO: check how to pass arguments in cross attention case.
            output = mod(query_emb, output, query_pos, input_pos, mask)
        return output


class CrossLinearTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        attn_type: Literal["fourier", "galerkin"],
        d_model: int,
        nhead: int,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation: Literal["relu", "gelu", "geglu"] = "relu",
        norm_first: bool = True,
        bias: bool = True,
        residual: bool = True,
        norm_config: NormlizationConfig = NormlizationConfig(method="InstanceNorm"),
        init_config: ParamInitConfig = ParamInitConfig(method="xavier", gain=1.0, diagonal_weight=1.0),
        pos_config: PosEncodingConfig = PosEncodingConfig(method="RoPE", dim=1, scale=1),
    ) -> None:
        super(CrossLinearTransformerEncoderLayer, self).__init__()

        self.norm_first: bool = norm_first
        self.dim_feedforward: Optional[int] = dim_feedforward
        self.residual: bool = residual

        self.norm1 = nn.LayerNorm(normalized_shape=d_model, bias=bias)
        self.cross_attn = CrossLinearMultiHeadAttention(
            attn_type=attn_type,
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            bias=bias,
            norm_config=norm_config,
            init_config=init_config,
            pos_config=pos_config,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model, bias=bias)
        if dim_feedforward is not None:
            self.feedforward = FeedForward(
                d_model=d_model,
                dim_feedforward=dim_feedforward,
                activation=activation,
                dropout=dropout,
                bias=bias,
            )

    def forward(
        self,
        query_emb: Tensor,
        input_emb: Tensor,
        query_pos: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.norm_first:
            input_emb = self.norm1(input_emb)
            output: Tensor = self.cross_attn(query_emb, input_emb, query_pos, input_pos, mask)
            output = self.norm2(output) + output if self.residual else self.norm2(output)
        else:
            output: Tensor = query_emb
            output = (
                self.cross_attn(query_emb, input_emb, query_pos, input_pos, mask) + output
                if self.residual
                else self.cross_attn(query_emb, input_emb, query_pos, input_pos, mask)
            )

        output = self.feedforward(output) + output if self.feedforward is not None else output

        return output
