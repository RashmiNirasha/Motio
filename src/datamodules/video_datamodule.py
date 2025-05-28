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

from src.datamodules.components.demo_dataset import Video_Dataset
from src.datamodules.configs import generate_data_pipeline


# ----------------------------------------------------------------------------
class VideoDataModule(LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
        train: bool = False,
        video_path="",
        path_to_pose=None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_predict: Optional[Dataset] = None

        self.data_class = Video_Dataset
        self.video_path = video_path
        self.path_to_pose = path_to_pose

    def prepare_data(self):
        """Download data if needed.
        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`."""
        # Predict
        if (stage == "predict") and (not self.data_predict):
            self.data_predict = self.data_class(
                self.hparams.cfg,
                split="test",
                transform_pipeline=generate_data_pipeline(self.hparams.cfg, "test"),
                video_path=self.video_path,
                path_to_pose=self.path_to_pose,
            )
        else:
            raise NotImplementedError("Implement predict dataset")

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.data_predict,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
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
