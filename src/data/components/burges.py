import os
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.io import loadmat
from torch import Tensor
from torch.utils.data import Dataset


class BURGERS(Dataset):
    """Burgers 1D equations dataset, from `https://github.com/neuraloperator/neuraloperator`

    :param root: Dataset location.
    :param x_data: Input function sampling value.
    :param y_data: Output function sampling value.
    :param pos: Input function sampling position.
    """

    def __init__(
        self,
        root: Union[str, Path],
        resolution: int = 2048,
        n_grid_total: int = 2**13,
        total_size: int = 1300,
    ) -> None:
        """__init__

        :param root: Dataset location, default to `data/`.
        :param resolution: Resolution of the dataset.
        :param n_grid_total: The number of grid points in dataset, default to 2**13.
        """
        self.root: Union[str, Path] = root
        subsampling_rate: int = n_grid_total // resolution

        raw_data: Dict[str, NDArray[np.float64]] = loadmat(
            file_name=os.path.join(self.raw_folder, "burgers_data_R10.mat")
        )
        x_data: NDArray[np.float64] = raw_data["a"][:, ::subsampling_rate]
        x_data = x_data[:total_size, :]
        y_data: NDArray[np.float64] = raw_data["u"][:, ::subsampling_rate]
        y_data = y_data[:total_size, :]

        self.x_data: Tensor = torch.as_tensor(
            x_data.reshape(total_size, resolution, 1),
            dtype=torch.float32,
        )
        self.y_data: Tensor = torch.as_tensor(
            y_data.reshape(total_size, resolution, 1),
            dtype=torch.float32,
        )
        self.input_pos: Tensor = torch.tensor(
            np.linspace(0, 1, resolution),
            dtype=torch.float32,
        ).reshape(resolution, 1)

        self.query_pos: Tensor = torch.tensor(
            np.linspace(0, 1, resolution),
            dtype=torch.float32,
        ).reshape(resolution, 1)

        self.gridx: Tensor = torch.reshape(
            torch.linspace(0, 1, resolution, dtype=torch.float32),
            (1, resolution, 1),
        )

    def __len__(self) -> int:
        assert len(self.x_data) == len(self.y_data)
        return len(self.x_data)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.x_data[idx, :, :], self.y_data[idx, :, :], self.input_pos, self.query_pos

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__)


if __name__ == "__main__":
    test_dataset = BURGERS(root="/home/jc/dev/lightning/data/")
    print(f"length of test_dataset: {len(test_dataset)}")

    x, y, pos1, pos2 = test_dataset.__getitem__(1)
    print(x.shape)
    print(y.shape)
    print(pos1.shape)
    print(pos2.shape)
