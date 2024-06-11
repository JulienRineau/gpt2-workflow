import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from gpt2 import GPT, GPTConfig, TextDataLoader


class GPTLightning(pl.LightningModule):
    def __init__(self, config: GPTConfig, batch_size: int):
        super().__init__()
        self.model = GPT(config)
        self.batch_size = batch_size
        self.save_hyperparameters()

    def forward(self, idx, targets=None):
        logits, loss = self.model(idx, targets)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self.model(x, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
        return optimizer

    def train_dataloader(self):
        return TextDataLoader(B=self.batch_size, T=32, num_workers=9)


if __name__ == "__main__":
    wandb_logger = WandbLogger(project="gpt2")

    batch_size = 4
    config = GPTConfig()
    model = GPTLightning(config, batch_size)

    wandb_logger.experiment.config["batch_size"] = batch_size

    trainer = pl.Trainer(max_epochs=50, accelerator="gpu", logger=wandb_logger)

    trainer.fit(model)
