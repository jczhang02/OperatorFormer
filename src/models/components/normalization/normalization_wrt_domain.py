from typing import Dict, Literal, Optional

from einops import rearrange
from torch import Tensor, nn
from torch.nn import InstanceNorm1d, LayerNorm

from .masked_instancenorm import MaskedInstanceNorm1d


__all__ = ["LayerNormWithWrtDomain", "InstanceNormWithWrtDomain"]


class LayerNormWithWrtDomain(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
    ) -> None:
        super(LayerNormWithWrtDomain, self).__init__()
        self.norm = LayerNorm(
            normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine, bias=bias
        )

    def forward(self, input: Tensor, mask: Optional[Tensor] = None, wrt_domain: bool = False) -> Tensor:
        # NOTE: Just use for oformer.
        if mask is not None:
            wrt_domain = False
        return (
            rearrange(
                self.norm(rearrange(input, "b h n d -> (b h) n d")),
                "(b h) n d -> b h n d",
                b=input.shape[0],
            )
            if wrt_domain
            else self.norm(input)
        )


class InstanceNormWithWrtDomain(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
    ) -> None:
        super(InstanceNormWithWrtDomain, self).__init__()

        self.norm: Dict[Literal["NoMask", "Masked"], nn.Module] = {
            "NoMask": InstanceNorm1d(
                num_features=num_features,
                eps=eps,
                momentum=momentum,
                affine=affine,
                track_running_stats=track_running_stats,
            ),
            # TODO: add MaskedInstanceNorm implementation.
            "Masked": MaskedInstanceNorm1d(
                num_features=num_features,
                eps=eps,
                momentum=momentum,
                affine=affine,
                track_running_stats=track_running_stats,
            ),
        }

    def forward(self, input: Tensor, mask: Optional[Tensor] = None, wrt_domain: bool = False) -> Tensor:
        # NOTICE: Instance Norm affine argument need to ensure `num_features` is equal to dim 1 of input.
        # But now, the input tensor has size (BH, N, D) and `num_features` is set as N.
        # Consider rearrange input tensor to size (BH, D, N).
        # Judge conditions:
        # 1. refer to performance,
        # 2. refer to galerkin transformer.
        if wrt_domain:
            return (
                rearrange(
                    self.norm["NoMask"](rearrange(input, "b h n d -> (b h) n d")),
                    "(b h) n d -> b h n d",
                    b=input.shape[0],
                )
                if mask is None
                else rearrange(
                    self.norm["Masked"](rearrange(input, "b h n d -> (b h) n d"), mask),
                    "(b h) n d -> b h n d",
                    b=input.shape[0],
                )
            )
        else:
            return self.norm["NoMask"](input) if mask is None else self.norm["Masked"](input, mask)
