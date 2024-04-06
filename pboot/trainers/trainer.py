from dataclasses import dataclass, asdict
from pathlib import Path
import os

import torch
import torch.nn as nn
from torch import Tensor

from pboot.config import set_config


@dataclass
class TrainerStats:
    train_loss: float = -1
    val_loss: float = -1
    test_loss: float = -1
    consumed_samples: int = 0


class PretrainingTrainer:
    def __init__(
        self,
        config,
        build_model_fn: callable,
        build_dataloader_fn: callable,
        build_optimizer_and_scheduler_fn: callable
    ) -> None:
        self.config = config
        self.build_model_fn = build_model_fn
        self.build_dataloader_fn = build_dataloader_fn
        self.build_optimizer_and_scheduler_fn = build_optimizer_and_scheduler_fn

        self.checkpoint_path = self._get_checkpoint_path(
            config.checkpoint_dir, config.checkpoint_prefix, config.checkpoint_format
        )

        if self._checkpoint_exists(
                self.checkpoint_path,
                config.checkpoint_format,
        ) is True:
            self._load(self.checkpoint_path)
        else:
            self._init()

        self.stats = TrainerStats()

    def _load(self, checkpoint_path: str):
        print(f"Loading checkpoint from path: {checkpoint_path}")
        state = torch.load(checkpoint_path)

        self.config = state.config
        set_config(self.config)

        self.dataloader = self.build_dataloader_fn(self.config)
        self.model = self.build_model_fn(self.config)
        self.model.load_state_dict(state.model_state)

        self.optim, self.scheduler = self.build_optimizer_and_scheduler_fn(
            self.config, self.model
        )
        self.optim.load_state_dict(state.optim_state)
        self.scheduler.load_scheduler_state(state.scheduler_state)
        self.stats = TrainerStats(state.stats)

    def _init(self):
        print("Initializing model from scratch")
        self.dataloader = self.build_dataloader_fn(self.config)
        self.model = self.build_model_fn(self.config)
        self.optim, self.scheduler = self.build_optimizer_and_scheduler_fn(
            self.config, self.model
        )

    @staticmethod
    def _get_checkpoint_path(ckpt_dir: str, ckpt_prefix: str, ckpt_format: str):
        if ckpt_format == "pt":
            ckpt_name = f"{ckpt_prefix}.pt"
            path = Path(ckpt_dir) / Path(ckpt_name)
        else:
            raise NotImplementedError
        return path

    @staticmethod
    def _checkpoint_exists(path, ckpt_format: str):
        if ckpt_format == "pt":
            return os.path.isfile(path)
        else:
            raise NotImplementedError

    def train_step(self, batch):
        self.optim.zero_grad()

        loss = self.model(batch)

        loss.backward()

        self.optim.step()
        self.scheduler.step()

        return loss.detach().clone()

    def val_step(self, batch):
        raise NotImplementedError

    def _train(self):
        self.model.train()
        batch_iter = iter(self.dataloader)

        interval_loss = 0.
        for i in range(1, self.config.train_steps + 1):
            # Sample from dataloader
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(self.dataloader)
                batch = next(batch_iter)
            batch = batch.to(self.config.device)

            # Train step
            interval_loss += self.train_step(batch)

            self.stats.consumed_samples += len(batch)

            # Logging loss metrics
            if i != 0 and i % self.config.loss_log_interval == 0:
                print(f"Loss step {i}: {interval_loss / self.config.loss_log_interval}")
                interval_loss = 0.

            # Validation steps
            if i != 0 and i % self.config.train_steps_per_val == 0:
                val_loss = self.validate()

            # Checkpoint interval
            if i != 0 and i % self.config.train_steps_per_checkpoint == 0:
                self.checkpoint()

    def train(self):
        if self.config.compile is True:
            train_fn = torch.compile(self._train)
            train_fn()
        else:
            self._train()

    def validate(self):
        self.model.eval()

        print("Validating")

        self.model.train()

    def checkpoint(self):
        checkpoint = PretrainingCheckpoint(
            self.checkpoint_path,
            self,
            self.config.checkpoint_format,
        )
        checkpoint.save()


class PretrainingCheckpoint:
    def __init__(self, ckpt_path: str, trainer: PretrainingTrainer, ckpt_format: str):
        self.ckpt_path = ckpt_path
        self.ckpt_format = ckpt_format

        self.model_state = trainer.model.state_dict()
        self.optim_state = trainer.optim.state_dict()
        self.scheduler_state = trainer.scheduler.state_dict()
        self.stats = asdict(trainer.stats)
        self.config = trainer.config

        self.state_dict = {
            "model": self.model_state,
            "optim": self.optim_state,
            "scheduler": self.scheduler_state,
            "stats": self.stats,
            "config": self.config,
        }

    def save(self):
        if self.ckpt_format == "pt":
            self._to_pt()
        else:
            raise NotImplementedError


    def _to_pt(self):
        dirpath = os.path.dirname(self.ckpt_path)
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath, exist_ok=False)

        torch.save(self.state_dict, self.ckpt_path)
