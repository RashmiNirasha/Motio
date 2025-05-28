#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arya Farkhondeh <arya.farkhondeh@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

from typing import Any, Dict, Optional

from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from src.datamodules.components.childplay_hand import ChildPlayHandDataset
from src.datamodules.components.childplay_hand_tas import ChildPlayHandDataset_TAS
from src.datamodules.configs import generate_data_pipeline


# ----------------------------------------------------------------------------
class ChildPlayHandDataModule(LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
        train: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.tasks = self.hparams.cfg.task.split("_")

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        if self.hparams.cfg.main_task == "recognition":
            self.data_class = ChildPlayHandDataset
        elif self.hparams.cfg.main_task == "segmentation":
            self.data_class = ChildPlayHandDataset_TAS
        else:
            raise

    def prepare_data(self):
        """Download data if needed.
        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`."""
        # Train
        if (stage == "fit") and (not self.data_train) and (not self.data_val):
            self.data_train = self.data_class(
                self.hparams.cfg,
                split="train",
                transform_pipeline=generate_data_pipeline(self.hparams.cfg, "train"),
            )
            print("Num. Trainign samples: ", len(self.data_train))
            self.data_val = self.data_class(
                self.hparams.cfg,
                split="val",
                transform_pipeline=generate_data_pipeline(self.hparams.cfg, "val"),
            )

        # Validation
        elif (stage == "validate") and (not self.data_val):
            self.data_val = self.data_class(
                self.hparams.cfg,
                split="val",
                transform_pipeline=generate_data_pipeline(self.hparams.cfg, "val"),
            )

        # Test
        elif (stage == "test") and (not self.data_test):
            self.data_test = self.data_class(
                self.hparams.cfg,
                split="test",
                transform_pipeline=generate_data_pipeline(self.hparams.cfg, "test"),
            )
        else:
            raise NotImplementedError("Implement predict dataset")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.cfg.train_batch_size,
            num_workers=self.hparams.cfg.train_num_workers,
            pin_memory=self.hparams.cfg.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.cfg.test_batch_size,
            num_workers=self.hparams.cfg.test_num_workers,
            pin_memory=self.hparams.cfg.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.cfg.test_batch_size,
            num_workers=self.hparams.cfg.test_num_workers,
            pin_memory=self.hparams.cfg.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    pass
