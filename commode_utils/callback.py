from os.path import split, join
from typing import Dict, List, Optional, Union

import torch
from pytorch_lightning import Callback, Trainer, LightningModule
from pytorch_lightning.loggers import WandbLogger

from commode_utils.common import print_table


class UploadCheckpointCallback(Callback):
    """Upload checkpoints after every validation epoch. Works only for wandb."""

    def __init__(self, checkpoint_dir: str):
        super().__init__()
        self._checkpoint_dir = checkpoint_dir

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        logger = trainer.logger
        if isinstance(logger, WandbLogger):
            experiment = logger.experiment
            root_dir, _ = split(self._checkpoint_dir)
            experiment.save(join(self._checkpoint_dir, "*.ckpt"), base_path=root_dir)


class PrintEpochResultCallback(Callback):
    """Print metrics to stdout after each specified epoch."""

    def __init__(
        self, after_train: bool = True, after_validation: bool = True, after_test: bool = True, split_symbol: str = "/"
    ):
        super().__init__()
        self.__after_train = after_train
        self.__after_validation = after_validation
        self.__after_test = after_test
        self.__split_symbol = split_symbol

    def _get_values(self, logged_metrics: Dict[str, Union[float, torch.Tensor]], step: str) -> List:
        values = []
        for key, value in logged_metrics.items():
            if self.__split_symbol not in key:
                continue
            group, metric = key.split(self.__split_symbol, 1)
            if group not in step:
                continue
            if isinstance(value, torch.Tensor):
                value = value.item()
            values.append(f"{metric} = {round(value, 2)}")
        return values

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule, unused: Optional[bool] = None):
        if not self.__after_train:
            return
        metric = {"train": self._get_values(trainer.logged_metrics, "train")}
        if f"val{self.__split_symbol}loss" in trainer.logged_metrics:
            metric["val"] = self._get_values(trainer.logged_metrics, "val")
        print_table(metric)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if not self.__after_validation:
            return
        metric = {"val": self._get_values(trainer.logged_metrics, "val")}
        print_table(metric)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if not self.__after_test:
            return
        metric = {"test": self._get_values(trainer.logged_metrics, "test")}
        print_table(metric)
