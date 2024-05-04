import torch
from torch import Tensor, nn


__all__ = ["PropagatorDecoder"]


class PropagatorDecoder(nn.Module):
    def __init__(self, d_model: int, out_features: int, num_layers: int, bias: bool = False) -> None:
        super(PropagatorDecoder, self).__init__()
        self.propagator = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model, bias=bias),
                    nn.GELU(),
                    nn.Linear(d_model, d_model, bias=bias),
                    nn.GELU(),
                    nn.Linear(d_model, d_model, bias=bias),
                )
                for _ in range(num_layers)
            ]
        )
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=bias),
            nn.GELU(),
            nn.Linear(d_model // 2, out_features, bias=True),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for block in self.propagator:
            for layers in block:
                for param in layers.parameters():
                    if param.ndim > 1:
                        in_c = param.size(-1)
                        nn.init.orthogonal_(param[:in_c], gain=1 / in_c)
                        param.data[:in_c] += 1 / in_c * torch.diag(torch.ones(param.size(-1), dtype=torch.float32))
                        if param.size(-2) != param.size(-1):
                            nn.init.orthogonal_(param[in_c:], gain=1 / in_c)
                            param.data[in_c:] += 1 / in_c * torch.diag(torch.ones(param.size(-1), dtype=torch.float32))

    def forward(self, z: Tensor) -> Tensor:
        for layer in self.propagator:
            z = layer(z) + z
        return self.out_proj(z)
