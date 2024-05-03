from dataclasses import dataclass, fields
from typing import List, Literal, Optional, Tuple, Union


__all__ = ["_unpack", "ParamInitConfig", "PosEncodingConfig", "NormlizationConfig"]


def _unpack(config) -> Tuple:
    return tuple(getattr(config, field.name) for field in fields(config))


# TODO: consider merge config validation checking function into dataclass.
@dataclass
class ParamInitConfig:
    method: Literal["xavier", "orthogonal"] = "orthogonal"
    gain: Optional[Union[int, float]] = None
    diagonal_weight: Optional[Union[int, float]] = None


@dataclass
class PosEncodingConfig:
    method: Literal["RoPE", "cat"] = "RoPE"
    dim: int = 1
    min_freq: float = 1 / 64
    scale: int = 1


@dataclass
class NormlizationConfig:
    method: Literal["LayerNorm", "InstanceNorm"] = "LayerNorm"
    eps: float = 1e-5
    elementwise_affine: bool = True
    affine: bool = False


@dataclass
class InputEncoderConfig:
    attn_type: Literal["fourier", "galerkin"] = "fourier"
    in_features: int = 1
    out_features: int = 96
    d_model: int = 96
    nhead: int = 8
    dim_feedforward: int = 96
    num_layers: int = 8
    scale: Union[int, List[int]] = 1
    dropout: float = 0.1
    activation: Literal["relu", "gelu", "geglu"] = "relu"
    norm_first: bool = True
    bias: bool = False
    norm_config: NormlizationConfig = NormlizationConfig(
        method="InstanceNorm",
        eps=1e-5,
        elementwise_affine=True,
        affine=False,
    )
    init_config: ParamInitConfig = ParamInitConfig(
        method="xavier",
        gain=1e-2,
        diagonal_weight=1e-4,
    )
    pos_config: PosEncodingConfig = PosEncodingConfig(
        method="RoPE",
        dim=1,
        min_freq=1 / 64,
        scale=1,
    )


@dataclass
class QueryEncoderConfig:
    in_features: int = 1
    d_model: int = 96
    bias: bool = False
    scale: int = 8


@dataclass
class CrossAttentionEncoderConfig:
    attn_type: Literal["fourier", "galerkin"] = "fourier"
    d_model: int = 96
    nhead: int = 8
    dim_feedforward: int = 96
    num_layers: int = 8
    scale: Union[int, List[int]] = 1
    dropout: float = 0.1
    activation: Literal["relu", "gelu", "geglu"] = "relu"
    norm_first: bool = True
    bias: bool = False
    residual: bool = True
    norm_config: NormlizationConfig = NormlizationConfig(
        method="InstanceNorm",
        eps=1e-5,
        elementwise_affine=True,
        affine=False,
    )
    init_config: ParamInitConfig = ParamInitConfig(
        method="xavier",
        gain=1e-2,
        diagonal_weight=1e-4,
    )
    pos_config: PosEncodingConfig = PosEncodingConfig(
        method="RoPE",
        dim=1,
        min_freq=1 / 64,
        scale=1,
    )


@dataclass
class PropagatorDecoderConfig:
    d_model: int = 96
    out_features: int = 1
    num_layers: int = 8
    bias: bool = False
