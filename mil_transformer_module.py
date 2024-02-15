import os
from argparse import ArgumentParser

import pandas as pd
import torch
import torchmetrics
from einops import rearrange
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.functional import auroc

from models.mil_transformer import MilTransformer


class MilTransformerModule(LightningModule):
    def __init__(
        self,
        bag_size,
        input_dim,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        dim_head,
        dropout=0.0,
        emb_dropout=0.0,
        optimizer="adamw",
        variant="vit",
        feature_extractor_model=None,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["feature_extractor_model"])

        self.model = MilTransformer(
            bag_size=bag_size,
            input_dim=input_dim,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
            variant=variant,
        )
        self.feature_extractor_model = feature_extractor_model
        self.optimizer = optimizer

        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")

    def training_step(self, batch, batch_idx):
        loss, scores, y = self.shared_step(batch)
        self.train_acc(scores, y)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc",
            self.train_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        return {"loss": loss, "scores": scores.detach(), "y": y}

    def training_epoch_end(self, outputs):
        self.log(
            "train_auc",
            auroc(
                torch.cat([x["scores"] for x in outputs], dim=0),
                torch.cat([x["y"] for x in outputs]),
                task="binary",
            ),
            prog_bar=True,
            logger=True,
        )

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self.shared_step(batch)
        self.val_acc(scores, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=False, logger=True)

        return {"loss": loss, "scores": scores, "y": y}

    def validation_epoch_end(self, outputs):
        self.log(
            "val_auc",
            auroc(
                torch.cat([x["scores"] for x in outputs], dim=0),
                torch.cat([x["y"] for x in outputs]),
                task="binary",
            ),
            prog_bar=True,
            logger=True,
        )

    def test_step(self, batch, batch_idx):
        features, patch_positions = batch["features"].squeeze(dim=0), batch[
            "tile locations"
        ].squeeze(dim=0)
        bag_dset = TensorDataset(features, patch_positions)
        # repeat and average the slide inference with different bag samples
        slide_scores = []
        for _ in range(self.hparams.test_repeats):
            bag_loader = DataLoader(
                bag_dset,
                batch_size=self.hparams.bag_size,
                shuffle=True,  # shuffle to get a random sample of patches
                num_workers=0,
                drop_last=True,
            )

            for bag, patch_positions in bag_loader:
                bag, patch_positions = bag.unsqueeze(dim=0), patch_positions.unsqueeze(
                    dim=0
                )
                logits = self.model((bag, patch_positions))
                score = logits.softmax(1)[0, 1].item()
                slide_scores.append(score)

        slide_score_max = torch.tensor(slide_scores, device=features.device).max()
        slide_score_avg = torch.tensor(slide_scores, device=features.device).mean()

        return {
            "slide_score_max": slide_score_max,
            "slide_score_avg": slide_score_avg,
            "y": batch["targets"],
            "slide_name": batch["slide name"],
            "slide_score_orig": batch["scores"],
        }

    def test_epoch_end(self, outputs):
        all_slide_scores_max = torch.stack([x["slide_score_max"] for x in outputs])
        all_slide_scores_avg = torch.stack([x["slide_score_avg"] for x in outputs])
        slide_targets = torch.stack([x["y"] for x in outputs]).squeeze()
        slide_names = [x["slide_name"][0] for x in outputs]
        slide_scores_orig = [x["slide_score_orig"].item() for x in outputs]

        # calculate auc if targets are not all the same (such as a default value in some test sets)
        if not slide_targets.unique().shape[0] == 1:
            self.log(
                "test_slide_auc_max",
                torchmetrics.functional.auroc(
                    all_slide_scores_max, slide_targets, task="binary"
                ),
            )
            self.log(
                "test_slide_auc_avg",
                torchmetrics.functional.auroc(
                    all_slide_scores_avg, slide_targets, task="binary"
                ),
            )

        # save slide scores
        df = pd.DataFrame(
            data={
                "Slide Name": slide_names,
                "MilTransformer Score AVG": all_slide_scores_avg.cpu(),
                "MilTransformer Score MAX": all_slide_scores_max.cpu(),
                "Regular Slide Score": slide_scores_orig,
                "Slide Label": slide_targets.cpu(),
            }
        )
        if isinstance(self.logger, WandbLogger):
            self.logger.log_table(key="slide_scores", dataframe=df)
            df.to_csv(os.path.join(self.logger.experiment.dir, "slide_scores.csv"))
        else:
            df.to_csv(os.path.join(self.logger.log_dir, "slide_scores.csv"))

        return

    def shared_step(self, batch):
        if self.feature_extractor_model is not None:
            data = batch["Data"]
            y = batch["Target"].squeeze()
            patch_positions = torch.zeros(size=(*data.shape, 2))
            with torch.no_grad():
                data = rearrange(data, "b1 b2 c h w -> (b1 b2) c h w")
                _, features = self.feature_extractor_model(data)
                features = rearrange(
                    features, "(b b2) e -> b b2 e", b2=self.hparams.bag_size
                )
        else:
            features = batch["features"]
            y = batch["targets"]
            patch_positions = batch["tile locations"]

        x = features
        logits = self.model((x, patch_positions))
        if logits.shape[0] == 1:  # handle batch size of 1 in patch dataset case
            y = y.unsqueeze(0)

        # get scores for positive class
        scores = logits.softmax(1)[:, 1]
        loss = F.cross_entropy(logits, y)

        return loss, scores, y

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        if self.optimizer == "sgd":
            optimizer = optim.SGD(
                self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=5e-4
            )
        else:
            optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[
                    int(0.3 * self.trainer.max_epochs),
                    int(0.7 * self.trainer.max_epochs),
                ],
                gamma=0.1,
            ),
            "interval": "epoch",
        }
        return [optimizer], [lr_scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--input_dim",
            type=int,
            default=512,
            help="dimension of patch input features",
        )
        parser.add_argument("--num_classes", type=int, default=2)
        parser.add_argument("--dim", type=int, default=512)
        parser.add_argument("--depth", type=int, default=2)
        parser.add_argument("--heads", type=int, default=8)
        parser.add_argument("--mlp_dim", type=int, default=512)
        parser.add_argument("--dim_head", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--emb_dropout", type=float, default=0.2)
        parser.add_argument(
            "--lr", "--learning_rate", dest="lr", type=float, default=0.01
        )
        parser.add_argument(
            "--optimizer", type=str, default="adamw", choices=["adamw", "sgd"]
        )
        parser.add_argument(
            "--variant", type=str, default="vit", choices=["simple", "vit"]
        )
        parser.add_argument("--test_repeats", type=int, default=10)
        return parser
