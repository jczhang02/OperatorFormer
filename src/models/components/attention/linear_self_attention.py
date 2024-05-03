from copy import deepcopy
from typing import Callable, Dict, List, Literal, Optional, Tuple, Type, cast

import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from ..activation import RotaryPositionalEncoding
from ..dataclass_config import NormlizationConfig, ParamInitConfig, PosEncodingConfig
from ..functional import _apply_rope_emb
from ..normalization import InstanceNormWithWrtDomain, LayerNormWithWrtDomain


__all__ = ["LinearMultiHeadAttention"]


class LinearMultiHeadAttention(nn.Module):
    def __init__(
        self,
        attn_type: Literal["fourier", "galerkin"],
        d_model: int,
        nhead: int = 8,
        dropout: float = 0.1,
        bias: bool = False,
        norm_config: NormlizationConfig = NormlizationConfig(
            method="LayerNorm",
            eps=1e-5,
            affine=False,
        ),
        init_config: ParamInitConfig = ParamInitConfig(
            method="xavier",
        ),
        pos_config: PosEncodingConfig = PosEncodingConfig(
            method="RoPE",
            dim=1,
            scale=1,
        ),
    ) -> None:
        super(LinearMultiHeadAttention, self).__init__()

        self._check_config_validation(
            attn_type=attn_type,
            norm_config=norm_config,
            init_config=init_config,
            pos_config=pos_config,
        )

        self.nhead: int = nhead
        self.d_model: int = d_model
        self.attn_type: Literal["fourier", "galerkin"] = attn_type
        self.qkv_projs: nn.ModuleList = nn.ModuleList(
            [deepcopy(nn.Linear(in_features=d_model, out_features=d_model * nhead, bias=bias)) for _ in range(3)]
        )
        self.norm_dict: Dict[str, nn.Module] = self._build_norm_dict(
            d_model=d_model, attn_type=attn_type, norm_config=norm_config
        )

        self._reset_parameters(init_config=init_config)

        self.pos_config: PosEncodingConfig = pos_config

        if pos_config.method == "RoPE":
            self.pos_encoding = RotaryPositionalEncoding(
                dim=d_model // pos_config.dim,
                min_freq=pos_config.min_freq,
                scale=pos_config.scale,
            )

        out_proj_use: bool = not (nhead == 1)

        self.out_proj = (
            nn.Sequential(
                nn.Linear(d_model * nhead + pos_config.dim * nhead, d_model, bias=bias),
                nn.Dropout(p=dropout),
            )
            if pos_config.method == "cat"
            else (
                nn.Sequential(
                    nn.Linear(d_model * nhead, d_model, bias=bias),
                    nn.Dropout(dropout),
                )
                if out_proj_use
                else nn.Identity()
            )
        )

    def forward(self, x: Tensor, pos: Optional[Tensor] = None, mask: Optional[Tensor] = None) -> Tensor:
        q: Tensor
        k: Tensor
        v: Tensor
        q, k, v = [rearrange(qkv_proj(x), "b n (h d) -> b h n d", h=self.nhead) for qkv_proj in self.qkv_projs]
        q, k, v, mask = self._normalization(q, k, v, mask)

        q, k, v = self._positional_encoding(q, k, v, pos)

        z: Tensor = self._attention_block(q, k, v, mask)

        return self.out_proj(z)

    def _normalization(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        if mask is None:
            # TODO: consider use False or True of mask value as the dict key of norm.
            if self.attn_type == "fourier":
                q = self.norm_dict["q"](q, wrt_domain=True)
                k = self.norm_dict["k"](k, wrt_domain=True)
            else:
                k = self.norm_dict["k"](k, wrt_domain=True)
                v = self.norm_dict["v"](v, wrt_domain=True)
        else:
            self.scale_factor = torch.sum(mask, dim=[-1, -2]).view(-1, 1, 1, 1)
            mask = repeat(mask, "b n d -> (b h) n d", h=self.nhead)

            if self.attn_type == "fourier":
                q = self.norm_dict["q"](q, mask=mask, wrt_domain=True)
                k = self.norm_dict["k"](k, mask=mask, wrt_domain=True)
            else:
                k = self.norm_dict["k"](k, mask=mask, wrt_domain=True)
                v = self.norm_dict["v"](v, mask=mask, wrt_domain=True)

            mask = rearrange(mask, "(b h) n d -> b h n d", h=self.nhead)

        return q, k, v, mask

    def _positional_encoding(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        pos: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        assert pos is not None, "`pos` is None. Please pass in `pos` argument."
        if self.pos_config is None:
            return q, k, v

        if self.pos_config.method == "RoPE":
            freqs: List[Tensor] = self.pos_encoding(pos)
            freqs = [repeat(freq, "b n d -> b h n d", h=self.nhead) for freq in freqs]
            q = _apply_rope_emb(q, freqs)
            k = _apply_rope_emb(k, freqs)
        elif self.pos_config.method == "cat":
            pos = repeat(pos, "b n d -> b h n d", h=self.nhead)
            q, k, v = [torch.cat([pos, input], dim=-1) for input in (q, k, v)]

        return q, k, v

    def _attention_block(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if mask is not None:
            q = q.masked_fill(mask, 0)
            k = k.masked_fill(mask, 0)
            v = v.masked_fill(mask, 0)
            attn_score: Tensor = k.transpose(-1, -2) @ v
            z: Tensor = (q @ attn_score) * (1.0 / self.scale_factor)
        else:
            attn_score: Tensor = k.transpose(-1, -2) @ v
            z: Tensor = (q @ attn_score) * (1.0 / q.shape[2])

        return rearrange(z, "b h n d -> b n (h d)")

    def _reset_parameters(self, init_config: ParamInitConfig) -> None:
        init_config.gain = 1 / self.d_model if init_config.gain is None else init_config.gain
        init_config.diagonal_weight = (
            1 / self.d_model if init_config.diagonal_weight is None else init_config.diagonal_weight
        )
        if init_config.method == "xavier":
            init_fn: Callable = nn.init.xavier_uniform_
        elif init_config.method == "orthogonal":
            init_fn: Callable = nn.init.orthogonal_

        for p in self.qkv_projs.parameters():
            if p.ndim > 1:
                init_fn(p, gain=cast(int, init_config.gain))
                if init_config.diagonal_weight > 0.0:
                    for h in range(self.nhead):
                        p.data[h * self.d_model : (h + 1) * self.d_model, :] += (
                            init_config.diagonal_weight * torch.diag(torch.ones(p.size(-1), dtype=torch.float))
                        )
            else:
                nn.init.constant_(p, 0)

    @staticmethod
    def _check_config_validation(
        attn_type: Literal["fourier", "galerkin"],
        norm_config: NormlizationConfig,
        init_config: ParamInitConfig,
        pos_config: PosEncodingConfig,
    ) -> None:
        alternative_attn_type_list: List[str] = ["fourier", "galerkin"]
        assert (
            attn_type in alternative_attn_type_list
        ), f"Alternative values for attention operation type are {alternative_attn_type_list}."

        alternative_norm_list: List[str] = ["LayerNorm", "InstanceNorm"]
        assert (
            norm_config.method in alternative_norm_list
        ), f"Alternative values for normalization method are {alternative_attn_type_list}."

        alternative_param_init_method_list: List[str] = ["xavier", "orthogonal"]
        assert (
            init_config.method in alternative_param_init_method_list
        ), f"Alternative values for parameter initialization method are {alternative_param_init_method_list}."

        alternative_posemb_list: List[str] = ["RoPE", "cat"]
        assert (
            pos_config.method in alternative_posemb_list
        ), f"Alternative values for positional encoding method are {alternative_posemb_list}."
        # TODO: add check to pos config and change config arguments name.

    @staticmethod
    def _build_norm_dict(
        d_model: int,
        attn_type: Literal["fourier", "galerkin"],
        norm_config: NormlizationConfig,
    ) -> Dict[str, nn.Module]:
        if norm_config.method == "LayerNorm":
            norm_cls: Type[nn.Module] = LayerNormWithWrtDomain
        else:
            norm_cls: Type[nn.Module] = InstanceNormWithWrtDomain

        # TODO: Consider how to process different arguments for LayerNormWithWrtDomain and InstanceNormWithWrtDomain,
        # add scalability for this function.
        if attn_type == "fourier":
            return {"q": norm_cls(d_model), "k": norm_cls(d_model)}
        else:
            return {"k": norm_cls(d_model), "v": norm_cls(d_model)}
