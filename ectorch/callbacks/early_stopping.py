from ectorch.callbacks.base import Callback


class EarlyStopping(Callback):
    """
    Early stopping callback.
    """
    def __init__(
        self,
        patience: int = 3,
        min_change: float = 0,
        target_metric: str = 'val_loss',
    ):
        """
        Initialize the early stopping callback.

        Args:
            target_metric (str): The metric to monitor for early stopping, default: val_loss.
            patience (int): Number of epochs with no improvement.
            min_change (float): Minimum change to qualify as an improvement.
        """
        super().__init__()
        self._patience = patience
        self._min_change = min_change
        self._target_metric = target_metric
        self._best_value = float('inf')
        self._epoch_count = 0

    def on_epoch_end(self, epoch: int):
        """Check for early stopping condition."""
        metric_value = self._trainer.history[self._target_metric][-1]
        if metric_value <= self._best_value - self._min_change:
            print(f'{metric_value=} {self._best_value=} {self._min_change=} {self._best_value-self._min_change}')
            self._best_value = metric_value
            self._epoch_count = 0
        else:
            self._epoch_count += 1

        if self._epoch_count >= self._patience:
            print(f'Early stopping at epoch {epoch + 1}.')
            self._trainer.stop_training()
