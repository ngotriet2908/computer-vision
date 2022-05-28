import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ClassificationDataset(Dataset):

    def __init__(self, path, train_fold=-1, img_size=512, mode='train', seed=2022):
        self.path = path
        self.mode = mode
        self.img_size = img_size

        if mode == 'test':
            self.csv = pd.read_csv(path + 'test/test_set.csv')
            all_image_names = np.array(self.csv[:]['Id'])
            all_labels = np.array(self.csv.drop(['Id'], axis=1))
        elif train_fold == -1:
            self.csv = pd.read_csv(path + 'train/train_set.csv')
            all_image_names = np.array(self.csv[:]['Id'])
            all_labels = np.array(self.csv.drop(['Id'], axis=1))
            train_imgs, val_imgs, train_labels, val_labels = train_test_split(
                all_image_names,
                all_labels,
                test_size=0.2,
                random_state=seed
            )
        else:
            self.csv = pd.read_csv(os.path.join(path,f'train/kfold/kfold_{train_fold}.csv'))

            all_image_names = np.array(self.csv[:]['Id'])
            all_labels = np.array(self.csv.drop(['Id'], axis=1))
            ratio = int(0.8 * len(all_image_names))

            train_imgs = all_image_names[:ratio]
            train_labels = all_labels[:ratio]
            val_imgs = all_image_names[ratio:]
            val_labels = all_labels[ratio:]

        if mode == 'train':
            self.image_names = train_imgs
            self.labels = train_labels
        elif mode == 'eval':
            self.image_names = val_imgs
            self.labels = val_labels
        elif mode == 'test':
            self.image_names = all_image_names
            self.labels = all_labels

        print("data {} length: {}".format(mode, len(self.image_names)))

        if mode == 'train':
            self.transform = A.Compose([
                A.ColorJitter(
                    brightness=0.5,
                    saturation=(0.7, 1.3),
                    contrast=(0.8, 1.2),
                    hue=0.1,
                    p=0.5,
                ),
                # A.ImageCompression(quality_lower=99, quality_upper=100, p=0.5),
                # A.GaussNoise(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Resize(self.img_size, self.img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                # A.CoarseDropout(p=0.5),
                ToTensorV2(),
            ],
                p=1.0,
            )
        elif mode == 'eval' or mode == 'test':
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        img_name = self.image_names[index]
        if self.mode == 'test':
            img_path = os.path.join(self.path, f'test/img_png/test_{img_name}.png')
        else:
            img_path = os.path.join(self.path, f'train/img_png/train_{img_name}.png')

        image = Image.open(img_path).convert("RGB")
        image = np.array(image).astype("uint8")
        image = self.transform(image=image)["image"]

        if self.mode == 'test':
            return {
                'img_id': img_name,
                'image': image
            }

        targets = self.labels[index]
        return {
            'img_id': img_name,
            'image': image,
            'label': torch.tensor(targets, dtype=torch.float32)
        }
