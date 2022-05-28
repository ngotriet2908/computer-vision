import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision import models as models
import torch.nn.functional as F

from src.classification.modelling.datasets import ClassificationDataset
from src.classification.modelling.metrics import dice


class ClassificationNetwork(pl.LightningModule):
    def __init__(self, path, batch_size=32, seed=2022, lr=1e-3, threshold=0.5, train_fold=-1, debug=False):
        super().__init__()
        self.model = models.resnet50(progress=True, pretrained=True)
        self.lr = lr
        self.debug = debug
        self.threshold = threshold
        self.batch_size = batch_size
        self.seed = seed
        self.train_fold = train_fold
        self.path = path

        for param in self.model.parameters():
            param.requires_grad = False

        # for checkpoint =< 9
        # self.model.fc = nn.Linear(2048, 20)

        # for checkpoint >= 10
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 20)
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def train_dataloader(self):
        train_dataset = ClassificationDataset(self.path, train_fold=self.train_fold, mode='train', seed=self.seed)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=10)

    def val_dataloader(self):
        val_dataset = ClassificationDataset(self.path, train_fold=self.train_fold, mode='eval', seed=self.seed)
        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=10)

    def test_dataloader(self):
        test_dataset = ClassificationDataset(self.path, mode='test', seed=self.seed)
        return DataLoader(test_dataset, batch_size=self.batch_size, num_workers=10)

    def training_step(self, batch, batch_idx):
        out = self(batch['image'])
        out = torch.sigmoid(out)
        labels = batch['label']
        loss = F.binary_cross_entropy(out, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch['image'])
        out = torch.sigmoid(out)
        labels = batch['label']
        loss = F.binary_cross_entropy(out, labels)
        self.log('val_loss', loss)
        return {
            'loss': loss.detach().cpu(),
            'label': batch['label'].detach().cpu(),
            'pred': out.detach().cpu()
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        out = self(batch['image'])
        out = torch.sigmoid(out)
        out = torch.nn.functional.threshold(out, self.threshold, 0.0)
        out = torch.ceil(out)
        return out.detach().cpu().numpy().astype(int)
        #
        # out = torch.round(out, decimals=3)
        # return out.detach().cpu().numpy()

    def training_epoch_end(self, training_step_outputs):
        outs = {
            k: [dic[k] for dic in training_step_outputs]
            for k in training_step_outputs[0]
        }
        losses = [x.item() for x in outs['loss']]
        avg_losses = np.mean(np.array(losses))
        if self.debug:
            print('training epoch avg_loss: {}'.format(avg_losses))

    def validation_epoch_end(self, validation_step_outputs):
        outs = {
            k: [dic[k] for dic in validation_step_outputs]
            for k in validation_step_outputs[0]
        }

        losses = [x.item() for x in outs['loss']]
        avg_losses = np.mean(np.array(losses))

        labels = torch.cat(outs['label']).numpy()

        preds = torch.cat(outs['pred']).numpy()
        preds[preds >= self.threshold] = 1.0
        preds[preds < self.threshold] = 0.0

        dice_scores = [dice(preds[i], labels[i]) for i in range(len(preds))]
        avg_dice_score = np.mean(np.array(dice_scores))

        if self.debug:
            print('-' * 80)
            print('epoch:{}'.format(self.current_epoch))
            print('validation epoch avg_loss: {}'.format(avg_losses))
            print('validation epoch dice score: {}'.format(avg_dice_score))

        self.log('val_avg_loss', avg_losses)
        self.log('val_avg_dice', avg_dice_score)
