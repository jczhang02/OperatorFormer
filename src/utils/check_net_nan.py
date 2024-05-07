import bcolors
import numpy as np
import rich
import torch


__all__ = ["check_net_value", "check_net_grad"]


def check_net_value(net) -> None:
    v_n = []
    v_v = []
    v_g = []
    for name, parameter in net.named_parameters():
        v_n.append(name)
        v_v.append(parameter.detach().cpu().numpy() if parameter is not None else [0])
        v_g.append(parameter.grad.detach().cpu().numpy() if parameter.grad is not None else [0])
    for i in range(len(v_n)):
        if np.max(v_v[i]).item() - np.min(v_v[i]).item() < 1e-6:
            color = bcolors.FAIL + "*"
        if np.isnan(v_v[i]).any():
            color = bcolors.FAIL + "*"
        else:
            color = bcolors.OK + " "
        print("%svalue %s: %.3e ~ %.3e" % (color, v_n[i], np.min(v_v[i]).item(), np.max(v_v[i]).item()))

        print("%sgrad  %s: %.3e ~ %.3e" % (color, v_n[i], np.min(v_g[i]).item(), np.max(v_g[i]).item()))


def check_net_grad(net) -> None:
    for name, parms in net.named_parameters():
        rich.print(
            "-->name:",
            name,
            "-->grad_requirs:",
            parms.requires_grad,
            "--weight",
            torch.mean(parms.data),
            " -->grad_value:",
            torch.mean(parms.grad),
        )
