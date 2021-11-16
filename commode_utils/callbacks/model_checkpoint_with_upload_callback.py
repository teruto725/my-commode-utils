from os import walk
from os.path import join

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


class ModelCheckpointWithUploadCallback(ModelCheckpoint):
    """Extend basic model checkpoint callback for immediate upload weights. Works only for wandb."""

    _CKPT_EXTENSION = ".ckpt"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._uploaded = []

    def save_checkpoint(self, trainer: Trainer) -> None:
        super().save_checkpoint(trainer)
        if not isinstance(trainer.logger, WandbLogger) or self.dirpath is None:
            return
        wandb_experiment = trainer.logger.experiment
        for root, _, files in walk(self.dirpath):
            for file in files:
                if not file.endswith(self._CKPT_EXTENSION):
                    continue
                if file in self._uploaded:
                    continue
                wandb_experiment.save(join(root, file), base_path=self.dirpath)
                self._uploaded.append(file)
