import logging
import math
import time
from dataclasses import dataclass

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from dataset import TextDataset
from gpt2 import GPT, GPTConfig, TextDataset


@dataclass
class TrainerConfig:
    batch_size: int = 4
    sequence_length: int = 1024
    max_lr_steps: int = 2000
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps = 200


class GPTLightning(pl.LightningModule):
    def __init__(
        self,
        model_config: GPTConfig,
        trainer_config: TrainerConfig,
    ):
        super().__init__()
        self.model = GPT(model_config)
        self.batch_size = trainer_config.batch_size
        self.sequence_length = trainer_config.sequence_length
        self.max_lr_steps = trainer_config.max_lr_steps
        self.max_lr = trainer_config.max_lr
        self.min_lr = trainer_config.min_lr
        self.warmup_steps = trainer_config.warmup_steps
        self.save_hyperparameters()

    def get_lr(self, it):
        if it < self.warmup_steps:
            return self.max_lr * (it + 1) / self.warmup_steps
        if it > self.max_lr_steps:
            return self.min_lr
        decay_ratio = (it - self.warmup_steps) / (self.max_lr_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    def forward(self, idx, targets=None):
        logits, loss = self.model(idx, targets)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            weight_decay=0.1,
            betas=[0.9, 0.95],
            eps=1e-8,
        )
        return optimizer

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure: bool = None,
    ):
        # Set the learning rate
        it = self.trainer.global_step
        lr = self.get_lr(it)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step(closure=optimizer_closure)
        # optimizer.zero_grad()

    def training_step(self, batch, batch_idx):
        start_time = time.time()  # Start timer

        x, y = batch
        logits, loss = self.model(x, y)

        # Logging
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)

        dt = time.time() - start_time
        tokens_per_sec = (self.batch_size * x.size(1)) / dt
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]

        self.log("dt", dt, on_step=True, logger=True)
        self.log("tokens_per_sec", tokens_per_sec, on_step=True, logger=True)
        self.log("learning_rate", current_lr, on_step=True, logger=True)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.trainer.global_step >= self.max_lr_steps:
            for param_group in self.trainer.optimizers[0].param_groups:
                param_group["lr"] = self.min_lr

    def train_dataloader(self):
        dataloader = DataLoader(
            TextDataset(sequence_length=self.sequence_length),
            batch_size=self.batch_size,
            shuffle=True,
        )
        return dataloader


if __name__ == "__main__":
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    wandb_logger = WandbLogger(project="gpt2")

    torch.set_float32_matmul_precision("high")

    model_config = GPTConfig(vocab_size=50304)
    trainer_config = TrainerConfig()
    model = GPTLightning(model_config, trainer_config)

    trainer = pl.Trainer(
        max_epochs=50,
        # max_steps=50,
        precision="bf16-mixed",
        logger=wandb_logger,
        gradient_clip_val=1.0,
        # accumulate_grad_batches=7,
        gradient_clip_algorithm="value",
        log_every_n_steps=1,
    )
    trainer.fit(model)

    wandb.finish()
