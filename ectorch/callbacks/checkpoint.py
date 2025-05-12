import os
from datetime import datetime

import torch

from ectorch.callbacks.base import Callback


class Checkpointer(Callback):
    """Checkpointing callback for saving model state."""
    def __init__(self, checkpoint_path: str, save_frequency: int = 1):
        super().__init__()
        self._checkpoint_path = checkpoint_path
        self._save_frequency = save_frequency

    def on_epoch_end(self, epoch: int):
        """Save the model checkpoint at the end of each epoch."""
        # TODO: Handle accelerate saves
        if not os.path.exists(self._checkpoint_path):
            os.makedirs(self._checkpoint_path)
        if (epoch + 1) % self._save_frequency == 0:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            save_path = os.path.join(
                self._checkpoint_path,
                f'checkpoint_{timestamp}_epoch_{epoch + 1}.pt',
            )
            torch.save(
                {
                    'epoch': epoch + 1,
                    'model_state_dict': self._trainer.model.state_dict(),
                    'optimizer_state_dict': self._trainer.optimizer.state_dict(),
                    'history': self._trainer.history,
                },
                save_path,
            )
            print(f'Model checkpoint saved to {save_path}.')
