from abc import abstractmethod
from os.path import split, join
from typing import Dict, List, Any

import torch
from pytorch_lightning import Callback, Trainer, LightningModule
from pytorch_lightning.loggers import WandbLogger

from commode_utils.common import print_table


class OldEpochEndCallback(Callback):
    """Implement behaviour of hooks like it was in <=1.1.7 version
    More details:
    https://github.com/PyTorchLightning/pytorch-lightning/issues/5917
    """

    def __init__(self):
        self._is_training = False

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        self._is_training = True

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
        self._is_training = False

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any):
        if trainer.val_dataloaders is None:
            self._old_on_epoch_end(trainer)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if self._is_training:
            self._old_on_epoch_end(trainer)

    @abstractmethod
    def _old_on_epoch_end(self, trainer: Trainer):
        raise NotImplementedError()


class UploadCheckpointCallback(OldEpochEndCallback):
    """Forced upload saved checkpoint after every epoch. Works only for wandb."""

    def __init__(self, checkpoint_dir: str):
        super().__init__()
        self._checkpoint_dir = checkpoint_dir

    def _old_on_epoch_end(self, trainer: Trainer):
        logger = trainer.logger
        if isinstance(logger, WandbLogger):
            experiment = logger.experiment
            root_dir, _ = split(self._checkpoint_dir)
            experiment.save(join(self._checkpoint_dir, "*.ckpt"), base_path=root_dir)


class PrintEpochResultCallback(OldEpochEndCallback):
    """Print metrics from specified groups after each epoch."""

    def __init__(self, *groups: str):
        super().__init__()
        self._groups = groups

    def _old_on_epoch_end(self, trainer: Trainer):
        metrics_to_print: Dict[str, List[str]] = {group: [] for group in self._groups}
        for key, value in trainer.callback_metrics.items():
            if "/" not in key:
                continue
            group, metric = key.split("/")
            if group in metrics_to_print:
                if isinstance(value, torch.Tensor):
                    value = value.item()
                metrics_to_print[group].append(f"{metric}={round(value, 2)}")
        print_table(metrics_to_print)
