from collections import OrderedDict
from typing import List, Literal, Optional, Tuple

import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from ..activation import PositionalEncoding
from ..functional import _get_activation_fn


__all__ = ["StandardMultiHeadAttention", "FeedForward"]


class StandardMultiHeadAttention(nn.Module):
    """Standard Multihead Attention from "Attention is all you need".

    References
    ----------
    Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[J].
    Advances in neural information processing systems, 2017, 30.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        bias: bool = False,
    ) -> None:
        """Initialize a StandardMultiHeadAttention class.

        Parameters
        ----------
        d_model : int
            The size of input tensor.
        nhead : int
            The number of head in attention.
        dropout : float
            The value p in Dropout.
        """
        super(StandardMultiHeadAttention, self).__init__()

        inner_dim: int = d_model * nhead
        self.scale: float = d_model**-0.5
        self.nhead: int = nhead

        self.qkv_proj = nn.Linear(
            d_model,
            inner_dim * 3,
            bias=bias,
        )
        self.softmax = nn.Softmax(dim=-1)
        self.pos_encoding = PositionalEncoding(d_model=d_model)
        self.out_proj = nn.Sequential(
            nn.Linear(
                inner_dim,
                d_model,
                bias=bias,
            ),
            nn.Dropout(p=dropout),
        )

    def forward(self, src: Tensor, pos: Optional[Tensor] = None, src_mask: Optional[Tensor] = None) -> Tensor:
        """Calculating the attention value z.

        Parameters
        ----------
        src : Tensor
            The input tensor with size ["batch_size", "sequence_length", "d_model"].
        src_mask : Optional[Tensor]
            The mask tensor with size ["batch_size", "sequence_length", "d_model"].

        Returns
        -------
        Tensor
            The calculated attention value z.
        """
        qkv_projed: Tensor = self.qkv_proj(src)
        qkv_projed_sep: Tuple[Tensor, ...] = torch.chunk(qkv_projed, chunks=3, dim=-1)

        q: Tensor
        k: Tensor
        v: Tensor
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.nhead) for t in qkv_projed_sep)

        if pos is not None:
            pos = self.pos_encoding(pos)

        scaled_dots: Tensor = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if src_mask is not None:
            src_mask_value = -1 * torch.finfo(scaled_dots.dtype).max
            src_mask = repeat(src_mask, "b n d -> b h n d", h=self.nhead)
            scaled_dots = scaled_dots.masked_fill(src_mask, src_mask_value)

        attn_value: Tensor = self.softmax(scaled_dots)

        z: Tensor = rearrange(torch.matmul(attn_value, v), "b h n d -> b n (h d)")
        z = self.out_proj(z)

        return z


class FeedForward(nn.Module):
    """FeedForward components in Transformer."""

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        activation: Literal["relu", "gelu", "geglu"] = "relu",
        dropout: float = 0.1,
        bias: bool = False,
    ) -> None:
        """Initialize a FeedForward class.

        Parameters
        ----------
        d_model : int
            The size of the input tensor.
        dim_feedforward : int
            The size of the tensor in feedforward components.
        activation : str
            Name of the expected activation function.
        dropout : float
            The value p in Dropout.
        """
        super(FeedForward, self).__init__()

        model_dict: OrderedDict[str, nn.Module] = OrderedDict(
            [
                ("fc1", nn.Linear(d_model, dim_feedforward, bias=bias)),
                ("activation", _get_activation_fn(activation=activation)),
                ("dropout1", nn.Dropout(dropout)),
                ("fc2", nn.Linear(dim_feedforward, d_model, bias=bias)),
                ("dropout2", nn.Dropout(dropout)),
            ],
        )

        if activation == "geglu":
            model_dict["fc1"] = nn.Linear(d_model, dim_feedforward * 2, bias=bias)

        self.feedforward = nn.Sequential(model_dict)

    def forward(self, x: Tensor) -> Tensor:
        """Forward method in FeedForward.

        Parameters
        ----------
        x : Tensor
            The input tensor with size [batch_size, sequence_length, d_model].

        Returns
        -------
        Tensor
            The output tensor with size [batch_size, sequence_length, d_model].
        """
        return self.feedforward(x)
