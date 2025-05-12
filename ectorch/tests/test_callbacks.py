import os
import shutil

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from ectorch.callbacks import Callback, Checkpointer, EarlyStopping
from ectorch.training import Trainer


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 1)

    def forward(self, x):
        return self.linear(x)


class TestCallback:
    def test_callback_base(self):
        callback = Callback()
        trainer = Trainer(SimpleModel(), optim.Adam(SimpleModel().parameters()), nn.MSELoss())
        callback.set_trainer(trainer)
        assert callback._trainer is trainer
        callback.on_train_start()
        callback.on_train_end()
        callback.on_epoch_start(0)
        callback.on_epoch_end(0)
        callback.on_train_batch_start(0)
        callback.on_train_batch_end(0)
        callback.on_val_batch_start(0)
        callback.on_val_batch_end(0)
        callback.on_predict_start()
        callback.on_predict_end()
        callback.on_predict_batch_start(0)
        callback.on_predict_batch_end(0)


class TestEarlyStopping:
    def setup_method(self):
        self.model = SimpleModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_func = nn.MSELoss()
        self.trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            loss_func=self.loss_func,
        )
        self.early_stopping = EarlyStopping(patience=3, min_change=0.01, target_metric='val_loss')
        self.early_stopping.set_trainer(self.trainer)

    def test_early_stopping(self):
        # Validation loss not improving
        losses = [0.1, 0.09, 0.08, 0.085, 0.09, 0.095]
        self.trainer._history['val_loss'] = []
        for epoch in range(len(losses)):
            self.trainer._history['val_loss'].append(losses[epoch])
            self.early_stopping.on_epoch_end(epoch)
        assert self.trainer._should_stop_training

    def test_early_stopping_with_improvement(self):
        # Validation loss improving
        losses = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05]
        self.trainer._history['val_loss'] = []
        for epoch in range(len(losses)):
            self.trainer._history['val_loss'].append(losses[epoch])
            self.early_stopping.on_epoch_end(epoch)
        assert not self.trainer._should_stop_training

    def test_early_stopping_min_change(self):
        # Validation loss improving, but not enough
        self.trainer._history['val_loss'] = [0.1, 0.0995, 0.099, 0.0985, 0.098, 0.0975]
        for epoch in range(len(self.trainer._history['val_loss'])):
            self.early_stopping.on_epoch_end(epoch)
        assert self.trainer._should_stop_training


class TestCheckpointer:
    @pytest.fixture(autouse=True)
    def checkpointer_setup(self, tmp_path):
        # Note: setup_method does not use fixtures, so we use a
        # custom fixture here for tmp_path
        self.checkpoint_path = tmp_path / 'checkpoints'
        self.model = SimpleModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_func = nn.MSELoss()
        self.trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            loss_func=self.loss_func,
        )
        self.checkpointer = Checkpointer(checkpoint_path=self.checkpoint_path, save_frequency=2)
        self.checkpointer.set_trainer(self.trainer)

    def teardown_method(self):
        # Remove the test checkpoint directory after the tests
        if os.path.exists(self.checkpoint_path):
            shutil.rmtree(self.checkpoint_path)

    def test_checkpointer(self):
        # Fake training loop
        num_epochs = 3
        for epoch in range(num_epochs):
            self.trainer._history['train_loss'].append(0.1)
            self.trainer._history['val_loss'].append(0.05)
            self.checkpointer.on_epoch_end(epoch)

        # Check number of checkpoints saved
        checkpoint_files = os.listdir(self.checkpoint_path)
        if num_epochs % self.checkpointer._save_frequency == 0:
            assert len(checkpoint_files) == num_epochs // self.checkpointer._save_frequency
        else:
            assert len(checkpoint_files) == num_epochs // self.checkpointer._save_frequency

        # Test loading a checkpoint
        checkpoint = checkpoint_files[-1]
        checkpoint_path = os.path.join(self.checkpoint_path, checkpoint)
        checkpoint = torch.load(checkpoint_path)
        assert isinstance(checkpoint['model_state_dict'], dict)
        assert isinstance(checkpoint['optimizer_state_dict'], dict)
        assert isinstance(checkpoint['history'], dict)
