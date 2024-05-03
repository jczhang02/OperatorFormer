from typing import Any, Dict, Optional, Tuple, cast

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from .components import BURGERS


class BurgersDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (1_000, 100, 200),
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
        resolution: int = 2048,
        n_grid_total: int = 2**13,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.total_size = train_val_test_split

        self.resolution: int = resolution
        self.n_grid_total: int = n_grid_total

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> None:
        pass

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:  # type: ignore
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."  # type: ignore
                )
            self.batch_size_per_device = (
                self.hparams.batch_size  # type: ignore
                // self.trainer.world_size
            )

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = BURGERS(
                self.hparams.data_dir,  # type: ignore
                resolution=self.resolution,
                n_grid_total=self.n_grid_total,
                total_size=sum(self.total_size),
            )
            # testset = BURGERS(
            #     self.hparams.data_dir,  # type: ignore
            #     resolution=self.resolution,
            #     n_grid_total=self.n_grid_total
            # )
            # dataset = ConcatDataset(datasets=[trainset, testset])
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,  # type: ignore
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=cast(Dataset, self.data_train),
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,  # type: ignore
            pin_memory=self.hparams.pin_memory,  # type: ignore
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=cast(Dataset, self.data_val),
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,  # type: ignore
            pin_memory=self.hparams.pin_memory,  # type: ignore
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=cast(Dataset, self.data_test),
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,  # type: ignore
            pin_memory=self.hparams.pin_memory,  # type: ignore
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = BurgersDataModule()
