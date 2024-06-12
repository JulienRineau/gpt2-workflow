import logging
import time

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from gpt2 import GPT, GPTConfig, TextDataset
from lightning.pytorch.utilities import grad_norm


class GPTLightning(pl.LightningModule):
    def __init__(self, config: GPTConfig, batch_size: int, sequence_length: int):
        super().__init__()
        self.model = GPT(config)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.save_hyperparameters()
        self.initial_lr = 3e-4
        self.min_lr = 3e-5

    def forward(self, idx, targets=None):
        logits, loss = self.model(idx, targets)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.initial_lr, weight_decay=0.1
        )
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,  # Number of iterations for the first restart
                T_mult=1,  # Factor to increase T_i after each restart
                eta_min=self.min_lr,  # Minimum learning rate
            ),
            "name": "lr_scheduler",
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        start_time = time.time()  # Start timer

        x, y = batch
        logits, loss = self.model(x, y)

        # Logging
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        dt = time.time() - start_time
        tokens_per_sec = (self.batch_size * x.size(1)) / dt
        self.log("dt", dt, on_step=True, logger=True)
        self.log("tokens_per_sec", tokens_per_sec, on_step=True, logger=True)

        return loss

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.layer, norm_type=2)
        self.log_dict(norms)

    def on_epoch_end(self):
        current_epoch = self.current_epoch
        if current_epoch >= 50:
            for param_group in self.optimizers().param_groups:
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

    batch_size = 4
    sequence_length = 32
    config = GPTConfig(vocab_size=50304)
    model = GPTLightning(config, batch_size, sequence_length)

    wandb_logger.experiment.config["batch_size"] = batch_size

    trainer = pl.Trainer(
        max_epochs=1,
        max_steps=50,
        precision="bf16-mixed",
        logger=wandb_logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=7,
        gradient_clip_algorithm="value",
    )
    trainer.fit(model)

    wandb.finish()
