import os
from typing import TYPE_CHECKING, Callable
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

try:
    from accelerate import Accelerator
    _HAS_ACCELERATE = True
except ImportError:
    warnings.warn(
        'accelerate not installed. If you want to use accelerate install it with `pip insall ectorch[accelerate]`',
    )
    _HAS_ACCELERATE = False

from ectorch.callbacks import Callback

if TYPE_CHECKING:
    from accelerate import Accelerator


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        device: torch.device | str = 'cpu',
        use_accelerate: bool = True,
        callbacks: list[Callback] | None = None,
    ):
        """A Trainer class for training PyTorch models."""
        self._model = model
        self._optimizer = optimizer
        self._loss_func = loss_func
        self._device = device

        self._should_stop_training = False

        self._history = {'train_loss': [], 'val_loss': []}

        self._accelerator: 'Accelerator | None' = None
        if use_accelerate and _HAS_ACCELERATE:
            self._accelerator = Accelerator()
            self._device = self._accelerator.device
            self._model, self._optimizer = self._accelerator.prepare(
                self._model,
                self._optimizer,
            )
            print(f'Using accelerate. Device is managed by accelerate device = {self._device}.')
        else:
            self._model.to(self._device)
            print(f'Not using accelerate. device = {self._device}.')

        self._callbacks: list[Callback]
        self._register_callbacks(callbacks)

    def _register_callbacks(self, callbacks: list[Callback] | None):
        self._callbacks = []
        if callbacks:
            for callback in callbacks:
                callback.set_trainer(self)
                self._callbacks.append(callback)

    # Properties
    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def optimizer(self) -> optim.Optimizer:
        return self._optimizer

    @property
    def loss_func(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        return self._loss_func

    @property
    def history(self) -> dict[str, list[float]]:
        return self._history

    def stop_training(self):
        """Stop training."""
        self._should_stop_training = True

    # Callback methods
    def _on_train_start(self):
        for callback in self._callbacks:
            callback.on_train_start()

    def _on_train_end(self):
        for callback in self._callbacks:
            callback.on_train_end()

    def _on_epoch_start(self, epoch: int):
        for callback in self._callbacks:
            callback.on_epoch_start(epoch)

    def _on_epoch_end(self, epoch: int):
        for callback in self._callbacks:
            callback.on_epoch_end(epoch)

    def _on_train_batch_start(self, batch_idx: int):
        for callback in self._callbacks:
            callback.on_train_batch_start(batch_idx)

    def _on_train_batch_end(self, batch_idx: int):
        for callback in self._callbacks:
            callback.on_train_batch_end(batch_idx)

    def _on_val_batch_start(self, batch_idx: int):
        for callback in self._callbacks:
            callback.on_val_batch_start(batch_idx)

    def _on_val_batch_end(self, batch_idx: int):
        for callback in self._callbacks:
            callback.on_val_batch_end(batch_idx)

    def _on_predict_start(self):
        for callback in self._callbacks:
            callback.on_predict_start()

    def _on_predict_end(self):
        for callback in self._callbacks:
            callback.on_predict_end()

    def _on_predict_batch_start(self, batch_idx: int):
        for callback in self._callbacks:
            callback.on_predict_batch_start(batch_idx)

    def _on_predict_batch_end(self, batch_idx: int):
        for callback in self._callbacks:
            callback.on_predict_batch_end(batch_idx)

    def _train_one_batch(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
    ) -> float:

        self._model.train()

        if not self._accelerator:
            x_batch, y_batch = x_batch.to(self._device), y_batch.to(self._device)

        self._optimizer.zero_grad()
        y_pred = self._model(x_batch)
        loss = self._loss_func(y_pred, y_batch)

        if self._accelerator:
            self._accelerator.backward(loss)
        else:
            loss.backward()

        self._optimizer.step()
        return loss.item()

    def _validate_one_batch(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
    ) -> float:

        self._model.eval()
        with torch.no_grad():
            if not self._accelerator:
                x_batch, y_batch = x_batch.to(self._device), y_batch.to(self._device)
            y_pred = self._model(x_batch)
            loss = self._loss_func(y_pred, y_batch)
        return loss.item()

    @staticmethod
    def _wrap_tensor(
        data: tuple[torch.Tensor, torch.Tensor] | DataLoader,
        batch_size: int = 32,
    ) -> DataLoader:
        """Wraps a tuple of tensors into a DataLoader."""
        if isinstance(data, DataLoader):
            return data
        else:
            return DataLoader(
                TensorDataset(*data),
                batch_size=batch_size,
            )

    def train(
        self,
        train_data: tuple[torch.Tensor, torch.Tensor] | DataLoader,
        *,
        val_data: tuple[torch.Tensor, torch.Tensor] | DataLoader | None = None,
        batch_size: int = 32,
        num_epochs: int = 1,
    ) -> dict[str, list[float]]:
        """
        Train the model.

        If train_data is a DataLoader, batch_size is ignored. Otherwise, train_data
        is wrapped in a TensorDataset then into a DataLoader with the specified
        batch size.

        Same for val_data.
        """
        self._should_stop_training = False
        self._on_train_start()

        train_loader = self._wrap_tensor(train_data, batch_size=batch_size)
        if val_data is not None:
            val_loader = self._wrap_tensor(val_data, batch_size=batch_size)
        else:
            val_loader = None

        if self._accelerator:
            train_loader = self._accelerator.prepare(train_loader)
            if val_loader:
                val_loader = self._accelerator.prepare(val_loader)

        for epoch in range(num_epochs):
            self._on_epoch_start(epoch)

            train_loss = 0.0
            val_loss = 0.0

            with tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
                for batch_idx, (x_train, y_train) in enumerate(pbar):
                    self._on_train_batch_start(batch_idx)
                    loss = self._train_one_batch(x_train, y_train)
                    train_loss += loss
                    self._on_train_batch_end(batch_idx)
                    pbar.set_postfix(loss=f'{loss:.4f}')

            train_loss_avg = train_loss / len(train_loader)
            self._history['train_loss'].append(train_loss_avg)
            print(f'Epoch {epoch + 1} Training Loss: {train_loss_avg:.4f}')

            if val_loader:
                with tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
                    for batch_idx, (x_val, y_val) in enumerate(pbar):
                        self._on_val_batch_start(batch_idx)
                        loss = self._validate_one_batch(x_val, y_val)
                        val_loss += loss
                        self._on_val_batch_end(batch_idx)
                        pbar.set_postfix(loss=f'{loss:.4f}')

                val_loss_avg = val_loss / len(val_loader)
                self._history['val_loss'].append(val_loss_avg)
                print(f'Epoch {epoch + 1} Validation Loss: {val_loss_avg:.4f}')
            self._on_epoch_end(epoch)

            if self._should_stop_training:
                print('Stopping training early.')
                break

        self._on_train_end()
        print('Done!')

        return self._history

    def _predict_one_batch(
        self,
        x_batch: torch.Tensor,
    ) -> torch.Tensor:
        self._model.eval()
        with torch.no_grad():
            if not self._accelerator:
                x_batch = x_batch.to(self._device)
            y_pred = self._model(x_batch)
        return y_pred

    def predict(
        self,
        x: torch.Tensor | DataLoader,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Predict using the model.

        If x is a DataLoader, batch_size is ignored. Otherwise, x is wrapped
        in a TensorDataset then into a DataLoader with the specified batch size.
        """
        self._on_predict_start()

        if isinstance(x, DataLoader):
            loader = x
        else:
            loader = DataLoader(
                TensorDataset(x),
                batch_size=batch_size,
            )

        if self._accelerator:
            loader = self._accelerator.prepare(loader)

        predictions = []
        with tqdm(loader, desc=f'Prediction', unit='batch') as pbar:
            for batch_idx, (x_batch,) in enumerate(pbar):
                self._on_predict_batch_start(batch_idx)
                y_pred = self._predict_one_batch(x_batch)
                predictions.append(y_pred.cpu())
                self._on_predict_batch_end(batch_idx)

        self._on_predict_end()

        return torch.cat(predictions, dim=0)

    def save(self, path: str | os.PathLike):
        """Save the model."""
        torch.save(
            {
                'model_state_dict': self._model.state_dict(),
                'optimizer_state_dict': self._optimizer.state_dict(),
                'history': self._history,
            },
            path,
        )

    def load(self, path: str | os.PathLike):
        """Load the model."""
        checkpoint = torch.load(path, map_location=self._device)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._history = checkpoint['history']
