import torch
from torch import nn

import rootutils

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from src.models.components import OperatorFormer
from src.models.components.dataclass_config import (
    InputEncoderConfig,
    QueryEncoderConfig,
    CrossAttentionEncoderConfig,
    PropagatorDecoderConfig,
)
from src.data.components.burges import BURGERS
import hydra
from src.models.components.loss import SimpleOperatorLearningL2Loss


def rel_loss(x, y, p, reduction=True, size_average=False, time_average=False):
    # x, y: [b, c, t, h, w] or [b, c, t, n]
    batch_num = x.shape[0]
    frame_num = x.shape[2]

    if len(x.shape) == 5:
        h = x.shape[3]
        w = x.shape[4]
        n = h * w
    else:
        n = x.shape[-1]
    # x = rearrange(x, 'b c t h w -> (b t h w) c')
    # y = rearrange(y, 'b c t h w -> (b t h w) c')
    num_examples = x.shape[0]
    eps = 1e-6
    diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), p, 1)
    y_norms = torch.norm(y.reshape(num_examples, -1), p, 1) + eps

    loss = torch.sum(diff_norms / y_norms)
    if reduction:
        loss = loss / batch_num
        if size_average:
            loss /= n
        if time_average:
            loss /= frame_num

    return loss


def central_diff(x: torch.Tensor, h):
    # assuming PBC
    # x: (batch, seq_len, feats), h is the step size

    pad_0, pad_1 = x[:, -2:-1], x[:, 1:2]
    x = torch.cat([pad_0, x, pad_1], dim=1)
    x_diff = (x[:, 2:] - x[:, :-2]) / 2  # f(x+h) - f(x-h) / 2h
    # pad = np.zeros(x_diff.shape[0])

    # return np.c_[pad, x_diff/h, pad]
    return x_diff / h


device = "cuda"


@hydra.main(version_base=None, config_path="./", config_name="test_for_train")
def train(cfg):
    dataset = BURGERS(root="/zhangchrai23/OperatorFormer/data", total_size=1100)
    train_dataset, test_dataset = random_split(dataset, [1000, 100])
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    model: nn.Module = hydra.utils.instantiate(cfg)
    model = model.to(device)
    loss_fn = SimpleOperatorLearningL2Loss()
    optim = torch.optim.Adam(
        model.parameters(),
        lr=1e-2,
    )
    sch = torch.optim.lr_scheduler.OneCycleLR(
        optim,
        max_lr=1e-2,
        total_steps=10000,
        div_factor=1e4,
        pct_start=0.2,
        final_div_factor=1e4,
    )

    train_epochs_loss = []

    for epoch in range(1000):
        model.train()
        epoch_loss = []

        for _, batch in enumerate(train_dataloader):
            x, y, input_pos, query_pos = batch
            x = x.to(device)
            y = y.to(device)
            input_pos = input_pos.to(device)
            query_pos = query_pos.to(device)

            pred = model(x, input_pos, query_pos)

            pred_loss = rel_loss(pred, y, 2)

            gt_d = central_diff(y, 1.0 / 2048)
            pred_d = central_diff(pred, 1.0 / 2048)
            d_loss = rel_loss(gt_d, pred_d, 2)

            print(f"pred_loss {pred_loss.item()}. deriv_loss {d_loss.item()}")

            loss = pred_loss + 1e-3 * d_loss

            optim.zero_grad()
            loss.backward()
            optim.step()
            sch.step()
            epoch_loss.append(loss.item())
            print(f"loss: {loss.item()}")


if __name__ == "__main__":
    train()
