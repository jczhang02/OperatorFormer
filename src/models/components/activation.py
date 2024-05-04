from typing import List

import torch
from einops import rearrange
from numpy import pi
from torch import Tensor, nn


__all__ = ["GaussianFourierFeatureTransform", "PositionalEncoding", "RotaryPositionalEncoding", "GEGLU"]


class GaussianFourierFeatureTransform(nn.Module):
    def __init__(self, num_input_channels: int, mapping_size: int = 256, scale=10) -> None:
        super(GaussianFourierFeatureTransform, self).__init__()

        self._num_input_channels: int = num_input_channels
        self._mapping_size: int = mapping_size
        self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size)) * scale, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        batches, _, _ = x.shape

        # Make shape compatible for matmul with _B.
        # From [B, N, C] to [(B*N), C].
        x = rearrange(x, "b n c -> (b n) c")

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = rearrange(x, "(b n) c -> b n c", b=batches)

        x = 2 * pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super(PositionalEncoding, self).__init__()
        self.d_model: int = d_model
        inv_freq: Tensor = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos: Tensor) -> Tensor:
        def get_emb(sin_inp: Tensor) -> Tensor:
            """
            Gets a base embedding for one dimension with sin and cos intertwined
            """
            emb: Tensor = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
            return torch.flatten(emb, -2, -1)

        assert len(pos.shape) == 3, "The input tensor `pos` need to be size [B, N, C]"
        batch_size, resolution, _ = pos.shape
        pos_x = torch.arange(resolution, device=pos.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((resolution, self.d_model), device=pos.device).type(pos.type())
        emb[:, : self.d_model] = emb_x

        return emb.repeat(batch_size, 1, 1)


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim: int, min_freq: float = 1 / 64, scale: float = 1.0) -> None:
        super(RotaryPositionalEncoding, self).__init__()
        inv_freq: Tensor = 1.0 / (10000 ** torch.arange(0, dim, 2, dtype=torch.float) / dim)
        self.register_buffer("inv_freq", inv_freq)
        self.min_freq: float = min_freq
        self.scale: float = scale

    def forward(self, pos: Tensor) -> List[Tensor]:
        pos_seps: List[Tensor] = [
            torch.squeeze(pos_sep, dim=-1) for pos_sep in torch.chunk(input=pos, chunks=pos.size(dim=-1), dim=-1)
        ]

        out: List[Tensor] = []

        for pos_sep in pos_seps:
            t: Tensor = pos_sep.type_as(self.inv_freq)
            t = t * (self.scale / self.min_freq)
            freqs: Tensor = torch.einsum("... i , j -> ... i j", t, self.inv_freq)
            out_sep: Tensor = torch.cat((freqs, freqs), dim=-1)
            out.append(out_sep)

        return out


class GEGLU(nn.Module):
    """GEGLU activation function.

    References
    ----------
        Shazeer et al., "GLU Variants Improve Transformer," 2020.
        "https://arxiv.org/abs/2002.05202".
    """

    def __init__(self) -> None:
        super(GEGLU, self).__init__()
        self.fn = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        """GEGLU activation forward

        Parameters
        ----------
        x : Tensor
            The input tensor with shape [B, H, W*2].

        Returns
        -------
        Tensor
            The calculated tensor with shape [B, H, W].
        """
        c = x.shape[-1]
        return self.fn(x[..., : int(c // 2)]) * x[..., int(c // 2) :]
