import numpy as np

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

from src.classification.modelling.models import ClassificationNetwork
from src.utilities.seed_everything import set_seed


def train(path, check_points_dir, train_fold=-1):
    set_seed(2022)

    model = ClassificationNetwork(path, train_fold=train_fold, lr=1e-3, threshold=0.5, debug=False)
    trainer = Trainer(
        devices=1,
        accelerator="auto",
        deterministic=True,
        max_epochs=100,
        log_every_n_steps=8,
        callbacks=[
            #  pl.callbacks.EarlyStopping(
            #     monitor="val_loss",
            #     mode="min",
            # ),
            ModelCheckpoint(
                save_top_k=2,
                monitor="val_avg_dice",
                mode="max",
                dirpath=check_points_dir,
                filename="checkpoint-cv3-classification-{epoch:02d}-{val_avg_dice:.3f}",
            )
        ]
    )
    trainer.fit(model)


def train_k_fold(path, check_points_dir, fold):
    train(path, check_points_dir, fold)


def train_all_fold(path, check_points_dir, folds=5):
    for i in range(folds):
        print("training fold {}".format(i))
        train(path, check_points_dir, i)


def predict():
    # PATH = '/content/drive/MyDrive/checkpoint-cv3-classification-epoch=83-val_avg_dice=0.798.ckpt'
    PATH = './checkpoint-cv3-classification-epoch=83-val_avg_dice=0.798.ckpt'

    model = ClassificationNetwork.load_from_checkpoint(PATH)
    trainer = Trainer(
        devices=1,
        accelerator="auto",
    )

    predictions = trainer.predict(model, dataloaders=model.test_dataloader())
    predictions_list = np.vstack(predictions)
    return predictions_list
