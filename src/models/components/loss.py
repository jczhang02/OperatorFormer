from typing import Tuple

import torch
from torch import Tensor, nn
from torchmetrics import Metric


__all__ = ["RelativeError", "SimpleOperatorLearningL2Loss"]


class RelativeError(Metric):
    def __init__(
        self,
        reduction: bool = True,
        eps: float = 1e-6,
        weight: Tuple[float, ...] = (1.0, 1e-3),
    ) -> None:
        """Initialization of L2Loss class.

        Parameters
        ----------
        reduction : bool
            Whether to perform mean loss on batch_size dimension.
        eps : float
            The eps value. Default is `1e-6`.
        """
        super().__init__()
        self.reduction: bool = reduction
        self.eps: float = eps
        self.weight: Tuple[float, ...] = weight
        self.add_state("relative_error", default=torch.tensor(0, dtype=torch.float64), dist_reduce_fx="mean")

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.relative_error = self._l2_norm(preds, target, reduction=self.reduction).to(torch.float64)

    def compute(self) -> Tensor:
        return self.relative_error

    @staticmethod
    def _l2_norm(pred: Tensor, gt: Tensor, reduction: bool = True, eps: float = 1e-6) -> Tensor:
        """Perform l2 norm with `pred` and `gt`.

        Parameters
        ----------
        pred : Tensor
            The output of model, i.e.,
            the expected sampling value of solution function derived from the proposed method.
        gt : Tensor
            The real ground truth, i.e.,
            the real sampling value of solution funcion derived from PDE solvers.

        Returns
        -------
        Tensor
            The calculated loss value.
        """
        num_examples: int = pred.shape[0]
        diff_norms: Tensor = torch.norm(
            input=pred.reshape(num_examples, -1) - gt.reshape(num_examples, -1),
            p=2,
            dim=1,
        )
        gt_norms: Tensor = (
            torch.norm(
                input=gt.reshape(num_examples, -1),
                p=2,
                dim=1,
            )
            + eps
        )
        loss: Tensor = torch.sum(diff_norms / gt_norms)

        if reduction:
            loss = loss / num_examples

        return loss


class SimpleOperatorLearningL2Loss(nn.Module):
    """Simple L2Loss for pde operator learning NNs.

    References
    ----------
    Copy from "https://github.com/BaratiLab/OFormer/blob/main/uniform_grids/loss_fn.py".
    """

    def __init__(
        self,
        reduction: bool = True,
        eps: float = 1e-6,
        weight: Tuple[float, ...] = (1.0, 1e-3),
    ) -> None:
        """Initialization of L2Loss class.

        Parameters
        ----------
        reduction : bool
            Whether to perform mean loss on batch_size dimension.
        eps : float
            The eps value. Default is `1e-6`.
        """
        super(SimpleOperatorLearningL2Loss, self).__init__()
        self.reduction: bool = reduction
        self.eps: float = eps
        self.weight: Tuple[float, ...] = weight

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        pred_loss: Tensor = self._l2_norm(pred, gt, reduction=self.reduction, eps=self.eps)
        d_loss: Tensor = self._l2_norm(
            pred=self._central_diff(pred, pred.shape[1]),
            gt=self._central_diff(gt, gt.shape[1]),
        )

        return self.weight[0] * pred_loss + self.weight[1] * d_loss

    @staticmethod
    def _l2_norm(pred: Tensor, gt: Tensor, reduction: bool = True, eps: float = 1e-6) -> Tensor:
        """Perform l2 norm with `pred` and `gt`.

        Parameters
        ----------
        pred : Tensor
            The output of model, i.e.,
            the expected sampling value of solution function derived from the proposed method.
        gt : Tensor
            The real ground truth, i.e.,
            the real sampling value of solution funcion derived from PDE solvers.

        Returns
        -------
        Tensor
            The calculated loss value.
        """
        num_examples: int = pred.shape[0]
        diff_norms: Tensor = torch.norm(
            input=pred.reshape(num_examples, -1) - gt.reshape(num_examples, -1),
            p=2,
            dim=1,
        )
        gt_norms: Tensor = (
            torch.norm(
                input=gt.reshape(num_examples, -1),
                p=2,
                dim=1,
            )
            + eps
        )
        loss: Tensor = torch.sum(diff_norms / gt_norms)

        if reduction:
            loss = loss / num_examples

        return loss

    @staticmethod
    def _central_diff(input: Tensor, resolution: int) -> Tensor:
        step_size: float = 1.0 / resolution
        pad_0: Tensor = input[:, -2:-1]
        pad_1: Tensor = input[:, 1:2]
        x: Tensor = torch.cat([pad_0, input, pad_1], dim=1)
        x_diff = (x[:, 2:] - x[:, :-2]) / 2

        return x_diff / step_size


if __name__ == "__main__":
    x1 = torch.randn(16, 2048, 1)
    x2 = torch.randn(16, 2048, 1)

    loss = SimpleOperatorLearningL2Loss()

    print(loss(x1, x2))
