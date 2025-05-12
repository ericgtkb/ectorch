from __future__ import annotations
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ectorch.training import Trainer


class Callback:
    """Base class for callbacks."""
    def __init__(self):
        self._trainer: Trainer

    def set_trainer(self, trainer: Trainer):
        """Set the trainer instance."""
        self._trainer = trainer

    def on_train_start(self):
        """Called at the beginning of training."""
        pass

    def on_train_end(self):
        """Called at the end of training."""
        pass

    def on_epoch_start(self, epoch: int):
        """
        Called at the beginning of each epoch.

        Note: epoch is 0-indexed.
        """
        pass

    def on_epoch_end(self, epoch: int):
        """
        Called at the end of each epoch.

        Note: epoch is 0-indexed.
        """
        pass

    def on_train_batch_start(self, batch_idx: int):
        """Called at the beginning of each training batch."""
        pass

    def on_train_batch_end(self, batch_idx: int):
        """Called at the end of each training batch."""
        pass

    def on_val_batch_start(self, batch_idx: int):
        """Called at the beginning of each validation batch."""
        pass

    def on_val_batch_end(self, batch_idx: int):
        """Called at the end of each validation batch."""
        pass

    def on_predict_start(self):
        """Called at the beginning of prediction."""
        pass

    def on_predict_end(self):
        """Called at the end of prediction."""
        pass

    def on_predict_batch_start(self, batch_idx: int):
        """Called at the beginning of each prediction batch."""
        pass

    def on_predict_batch_end(self, batch_idx: int):
        """Called at the end of each prediction batch."""
        pass
