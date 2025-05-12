import os

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ectorch.callbacks import Callback
from ectorch.training import Trainer


class SimpleModel(nn.Module):
    def __init__(self, input_size=16):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 4)
        self.linear2 = nn.Linear(4, 1)

    def forward(self, x):
        return self.linear2(self.linear1(x))


class TestTrainer:
    def setup_method(self):
        self.input_size = 16
        self.model = SimpleModel(input_size=self.input_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_func = nn.MSELoss()
        self.train_data = (torch.randn(100, self.input_size), torch.randn(100, 1))
        self.val_data = (torch.randn(50, self.input_size), torch.randn(50, 1))
        self.trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            loss_func=self.loss_func,
        )

    def test_trainer_init(self):
        assert isinstance(self.trainer._model, nn.Module)
        assert isinstance(self.trainer._optimizer, optim.Optimizer)
        assert callable(self.trainer._loss_func)
        assert self.trainer._history == {'train_loss': [], 'val_loss': []}

    def test_trainer_init_device(self):
        trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            loss_func=self.loss_func,
            device='cpu',
            use_accelerate=False,
        )
        assert trainer._device == 'cpu'

        trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            loss_func=self.loss_func,
            device='cuda',
            use_accelerate=False,
        )

        assert trainer._device == 'cuda'

    def test_properties(self):
        assert isinstance(self.trainer.model, nn.Module)
        assert isinstance(self.trainer.optimizer, optim.Optimizer)
        assert callable(self.trainer.loss_func)
        assert isinstance(self.trainer.history, dict)

    def test_stop_training(self):
        assert not self.trainer._should_stop_training
        self.trainer.stop_training()
        assert self.trainer._should_stop_training

    def test_register_callbacks(self):
        callback1 = Callback()
        callback2 = Callback()
        callbacks = [callback1, callback2]
        self.trainer._register_callbacks(callbacks)
        assert len(self.trainer._callbacks) == 2
        assert self.trainer._callbacks[0] is callback1
        assert self.trainer._callbacks[1] is callback2
        assert callback1._trainer is self.trainer
        assert callback2._trainer is self.trainer

    def test_callback_methods(self, mocker):
        mock_callback = mocker.MagicMock()
        self.trainer._callbacks = [mock_callback]

        self.trainer._on_train_start()
        mock_callback.on_train_start.assert_called_once()

        self.trainer._on_train_end()
        mock_callback.on_train_end.assert_called_once()

        self.trainer._on_epoch_start(0)
        mock_callback.on_epoch_start.assert_called_once_with(0)

        self.trainer._on_epoch_end(0)
        mock_callback.on_epoch_end.assert_called_once_with(0)

        self.trainer._on_train_batch_start(0)
        mock_callback.on_train_batch_start.assert_called_once_with(0)

        self.trainer._on_train_batch_end(0)
        mock_callback.on_train_batch_end.assert_called_once_with(0)

        self.trainer._on_val_batch_start(0)
        mock_callback.on_val_batch_start.assert_called_once_with(0)

        self.trainer._on_val_batch_end(0)
        mock_callback.on_val_batch_end.assert_called_once_with(0)

        self.trainer._on_predict_start()
        mock_callback.on_predict_start.assert_called_once()

        self.trainer._on_predict_end()
        mock_callback.on_predict_end.assert_called_once()

        self.trainer._on_predict_batch_start(0)
        mock_callback.on_predict_batch_start.assert_called_once_with(0)

        self.trainer._on_predict_batch_end(0)
        mock_callback.on_predict_batch_end.assert_called_once_with(0)

    def test_train_one_batch(self):
        x_batch = torch.randn(32, self.input_size).to(self.trainer.device)
        y_batch = torch.randn(32, 1).to(self.trainer.device)
        loss = self.trainer._train_one_batch(x_batch, y_batch)
        assert isinstance(loss, float)

    def test_validate_one_batch(self):
        x_batch = torch.randn(32, self.input_size).to(self.trainer.device)
        y_batch = torch.randn(32, 1).to(self.trainer.device)
        loss = self.trainer._validate_one_batch(x_batch, y_batch)
        assert isinstance(loss, float)

    def test_wrap_tensor_with_dataloader(self):
        dataloader = DataLoader(TensorDataset(*self.train_data), batch_size=32)
        wrapped_loader = self.trainer._wrap_tensor(dataloader)
        assert wrapped_loader is dataloader

    def test_wrap_tensor_with_tensors(self):
        dataloader = self.trainer._wrap_tensor(self.train_data, batch_size=128)
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 128

    def test_train(self, mocker):
        # Patch tqdm to not actually print progress bars
        mock_tqdm = mocker.patch('tqdm.auto.tqdm')
        mock_tqdm.return_value.__iter__.return_value = [(torch.randn(32, self.input_size), torch.randn(32, 1))]
        history = self.trainer.train(
            train_data=self.train_data,
            val_data=self.val_data,
            num_epochs=1,
        )
        assert isinstance(history, dict)
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 1
        assert len(history['val_loss']) == 1

    def test_train_dataloader(self, mocker):
        # Patch tqdm to not actually print progress bars
        mock_tqdm = mocker.patch('tqdm.auto.tqdm')
        mock_tqdm.return_value.__iter__.return_value = [(torch.randn(32, self.input_size), torch.randn(32, 1))]
        train_loader = DataLoader(TensorDataset(*self.train_data), batch_size=32)
        val_loader = DataLoader(TensorDataset(*self.val_data), batch_size=32)
        history = self.trainer.train(
            train_data=train_loader,
            val_data=val_loader,
            num_epochs=1,
        )
        assert isinstance(history, dict)
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 1
        assert len(history['val_loss']) == 1

    def test_predict_one_batch(self):
        x_batch = torch.randn(32, self.input_size).to(self.trainer.device)
        predictions = self.trainer._predict_one_batch(x_batch)
        assert predictions.shape == (32, 1)

    def test_predict(self, mocker):
        # Patch tqdm to not actually print progress bars
        mock_tqdm = mocker.patch('tqdm.auto.tqdm')
        mock_tqdm.return_value.__iter__.return_value = [(torch.randn(32, self.input_size),)]
        x_test = torch.randn(100, self.input_size)
        predictions = self.trainer.predict(x_test)
        assert predictions.shape == (100, 1)

    def test_predict_dataloader(self, mocker):
        mock_tqdm = mocker.patch('tqdm.auto.tqdm')
        mock_tqdm.return_value.__iter__.return_value = [(torch.randn(32, self.input_size),)]
        test_loader = DataLoader(TensorDataset(torch.randn(100, self.input_size)), batch_size=32)
        predictions = self.trainer.predict(test_loader)
        assert predictions.shape == (100, 1)

    def test_save_load(self, tmp_path):
        # Create a temporary file for saving and loading
        temp_file = tmp_path / 'tmp_model.pt'
        self.trainer.save(temp_file)
        self.trainer.load(temp_file)
        # Clean up the temporary file
        os.remove(temp_file)
