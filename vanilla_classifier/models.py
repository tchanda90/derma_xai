import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
cv2.setNumThreads(1)
import wandb
import pytorch_lightning as pl
import albumentations
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch.utils.data.sampler import WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from albumentations.pytorch import ToTensorV2
from captum.attr import LayerAttribution, LayerGradCam
from torchmetrics import Accuracy, AUROC, Recall, Specificity
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from sklearn import metrics
from sklearn.model_selection import train_test_split
from dataset import *
num_workers = 16



def get_transforms(image_size, full=False):

    if full:
        transforms_train = albumentations.Compose([
            albumentations.Transpose(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.ColorJitter(p=0.5),
            albumentations.OneOf([
                albumentations.MotionBlur(blur_limit=5),
                albumentations.MedianBlur(blur_limit=5),
                albumentations.GaussianBlur(blur_limit=(3, 5)),
                albumentations.GaussNoise(var_limit=(5.0, 30.0)),
            ], p=0.7),
            albumentations.OneOf([
                albumentations.OpticalDistortion(distort_limit=1.0),
                albumentations.GridDistortion(num_steps=5, distort_limit=1.),
                albumentations.ElasticTransform(alpha=3),
            ], p=0.7),
            albumentations.CLAHE(clip_limit=4.0, p=0.7),
            albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            albumentations.Resize(image_size, image_size),
            albumentations.CoarseDropout(max_height=int(image_size * 0.375), max_width=int(image_size * 0.375), max_holes=1, p=0.3),
            albumentations.Normalize(),
            ToTensorV2()
        ])
    else:
        transforms_train = albumentations.Compose([
            albumentations.Transpose(p=0.2),
            albumentations.VerticalFlip(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.ColorJitter(p=0.5),
            albumentations.CLAHE(clip_limit=4.0, p=0.7),
            albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            albumentations.CoarseDropout(max_height=int(image_size * 0.1), max_width=int(image_size * 0.1), max_holes=1, p=0.3),
            albumentations.Resize(image_size, image_size),
            albumentations.Normalize(),
            ToTensorV2()
        ])

    transforms_test = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize(),
        ToTensorV2()
    ])
    return transforms_train, transforms_test


class BaselineClassifier(pl.LightningModule):
    def __init__(self, hidden_size=64, learning_rate=2e-4, img_dir='./', metadata_file='./', train_data_dir='./', val_data_dir='./',
                 test_data_dir='./', batch_size=32, dropout=0.5, weighted_sampling=False, dx_pos_weight=dx_pos_weight):

        super().__init__()
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.metadata_file = metadata_file
        self.weighted_sampling = weighted_sampling
        self.learning_rate = learning_rate
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.dropout = dropout
        
        self.train_set, self.val_set, self.test_set, self.external_set = None, None, None, None
        
        self.loss = nn.BCEWithLogitsLoss(pos_weight=dx_pos_weight)

        self.train_transform, self.test_transform = get_transforms(image_size=image_size)
        
        self.num_classes=1
        base_model = models.resnet50(pretrained=True)
        base_model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=base_model.fc.in_features, out_features=self.num_classes),
        )
        
        self.base_model = base_model

        self.layer_gc = LayerGradCam(self.base_model, self.base_model.layer4[-1])

        self.sigmoid = nn.Sigmoid()

        self.accuracy = Accuracy(threshold=0.5, average='macro', num_classes=self.num_classes)
        self.auroc = AUROC(average='macro', num_classes=self.num_classes)
        self.sensitivity = Recall(threshold=0.5, average='macro', num_classes=self.num_classes)
        self.specificity = Specificity(threshold=0.5, average='macro', num_classes=self.num_classes)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            'scheduler': CosineAnnealingLR(optimizer, T_max=num_epochs, verbose=False),
            'interval': 'epoch',
            'frequency': 1
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    @staticmethod
    def inverse_normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor
    
    def forward(self, x, zero_non_predicted=True):
        
        output = self.base_model(x)

        return output

    def on_train_start(self):

        self.log("hp/attention_weight", float(attention_weight))
        self.log("hp/lr", learning_rate)
        self.log("hp/batch_size", float(batch_size))
        self.log("hp/dropout", dx_dropout)
        self.log("hp/weighted_sampling", float(weighted_sampling))

    def training_step(self, batch, batch_idx):
        x, y = batch
                        
        y_dx, y_char, y_annotations, image_name = y
        diagnosis_predictions = self(x)
        loss = self.loss(diagnosis_predictions, y_dx)        
        
        self.log("train/loss", loss, on_epoch=True, on_step=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        
        diagnosis_predictions = self(x)
        loss = self.loss(diagnosis_predictions, y_dx)       
                
        self.log("val/loss", loss, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        
        diagnosis_predictions = self(x)
        loss = self.loss(diagnosis_predictions, y_dx)       

        self.accuracy(output.round(), y_char.int())
        self.auroc(output, y_char.int())
        self.sensitivity(output, y_char.int())
        self.specificity(output, y_char.int())

    def on_test_epoch_end(self) -> None:
        self.log("test/bal_acc", self.accuracy.compute())
        self.log("test/auroc", self.auroc.compute())
        self.log("test/sensitivity", self.sensitivity.compute())
        self.log("test/specificity", self.specificity.compute())

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        
        diagnosis_predictions = self(x)
        loss = self.loss(diagnosis_predictions, y_dx)       
        
        return diagnosis_predictions, y_dx, image_name, self.inverse_normalize(x)

    def setup(self, stage=None):
        if stage == 'fit':
            metadata = pd.read_csv(metadata_file)
            
            train_set = metadata[metadata.split == 'train'] 
            val_set = metadata[metadata.split == 'val']
            test_set = metadata[metadata.split == 'test']
            external_set = metadata[metadata.split == 'external']
                        
            dist = train_set[dx_class_label].value_counts()
            pos_weight = [dist[0]/dist[1]]
            self.loss.pos_weight = torch.Tensor(pos_weight)

            self.train_set = MelanomaCharacteristicsDataset(img_dir=self.img_dir,
                                                            annotations_dir=self.annotations_dir,
                                                            metadata=train_set,
                                                            transform=self.train_transform)

            self.val_set = MelanomaCharacteristicsDataset(img_dir=self.img_dir,
                                                          annotations_dir=self.annotations_dir,
                                                          metadata=val_set,
                                                          transform=self.test_transform)

            self.test_set = MelanomaCharacteristicsDataset(img_dir=self.img_dir,
                                                           annotations_dir=self.annotations_dir,
                                                           metadata=test_set,
                                                           transform=self.test_transform)
            
            self.external_set = MelanomaCharacteristicsDataset(img_dir=self.img_dir,
                                                           annotations_dir=self.annotations_dir,
                                                           metadata=external_set,
                                                           transform=self.test_transform)
            
            
    

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def external_dataloader(self):
        return DataLoader(self.external_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)


    
    
    
    