import torch
import numpy as np
from rich import print

x = torch.randn(16, 2048, 1)
gridx = torch.tensor(np.linspace(0, 1, 2048), dtype=torch.float32)

gridx = gridx.reshape(1, 2048, 1)
print(gridx.shape)
pos = gridx.repeat([x.shape[0], 1, 1])
print(x.shape)
print(pos.shape)
cat_x = torch.cat((x, pos), dim=-1)
print(cat_x.shape)

x = torch.randn(16, 2048, 96)
q = torch.randn(16, 8, 2048, 96)
freqs_x = torch.randn(16, 8, 2048, 96)
freqs_y = torch.randn(16, 8, 2048, 96)

from einops import rearrange, repeat


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


def apply_2d_rotary_pos_emb(t, freqs_x, freqs_y):
    # split t into first half and second half
    # t: [b, h, n, d]
    # freq_x/y: [b, n, d]
    d = t.shape[-1]
    t_x, t_y = t[..., : d // 2], t[..., d // 2 :]

    return torch.cat((apply_rotary_pos_emb(t_x, freqs_x), apply_rotary_pos_emb(t_y, freqs_y)), dim=-1)


apply_2d_rotary_pos_emb(q, freqs_x, freqs_y)
