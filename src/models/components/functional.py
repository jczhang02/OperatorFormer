import copy
from typing import List, Literal

import torch
from einops import rearrange
from torch import Tensor, nn

from .activation import GEGLU


__all__ = ["_get_clones", "_apply_rope_emb", "_get_activation_fn"]


def _get_clones(module: nn.Module, num_layers: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_layers)])


def _apply_rope_emb(input: Tensor, freqs: List[Tensor]) -> Tensor:
    assert len(freqs) <= 2, f"`freqs` has size {len(freqs)}, but RoPE only supports 1D and 2D cases."
    d: int = input.shape[-1]
    pos_dim: int = len(freqs)
    t_components: List[Tensor] = []
    pos_emb: List[Tensor] = []

    def rotate_half(input: Tensor) -> Tensor:
        x = rearrange(input, "... (j d) -> ... j d", j=2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def single_rope_emb_process(t: Tensor, freqs: Tensor) -> Tensor:
        return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())

    for idx, _ in enumerate(freqs):
        t_components.append(input[..., idx * (d // pos_dim) : (idx + 1) * (d // pos_dim)])

    for idx in range(pos_dim):
        pos_emb.append(single_rope_emb_process(t_components[idx], freqs[idx]))

    return torch.cat((pos_emb), dim=-1)


def _get_activation_fn(activation: Literal["relu", "gelu", "geglu"]) -> nn.Module:
    """Get activation callable object from str.
       Notice: Using nn.Module to ensure parameters in activation functions
            are included in overall network instead of nn.functional.

    Parameters
    ----------
    activation : str
        Name of the expected activation function, should be `relu`, `gelu` or `geglu`.

    Returns
    -------
    nn.Module
        Callable of the expected activation function.
    """
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "geglu":
        return GEGLU()
    else:
        raise RuntimeError(f"activation should be relu/gelu/geglu, not {activation}")


if __name__ == "__main__":
    print(_apply_rope_emb(input=torch.randn(16, 2048, 96), freqs=[torch.randn(16, 2048, 96)]))
    print(_get_activation_fn(activation="relu"))
