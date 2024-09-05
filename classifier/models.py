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


class DiceLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets, smooth=1):
        
        num = targets.shape[0]
        inputs = inputs.reshape(num, -1)
        targets = targets.reshape(num, -1)

        intersection = (inputs * targets).sum(1)
        dice = (2. * intersection + smooth) / (inputs.sum(1) + targets.sum(1) + smooth)
        
        if self.reduction == 'mean':
            dice = dice.sum() / num

        return 1 - dice
    

    

class CharacteristicsClassifier(pl.LightningModule):
    def __init__(self, hidden_size=64, learning_rate=2e-4, img_dir='./', annotations_dir='./', metadata_file='./', train_data_dir='./',
                 val_data_dir='./', test_data_dir='./', batch_size=32, dx_dropout=0.5, char_dropout=0.5, weighted_sampling=False,
                 char_pos_weight=char_pos_weight, dx_pos_weight=dx_pos_weight):

        super().__init__()
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.metadata_file = metadata_file
        
        self.weighted_sampling = weighted_sampling
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.dx_dropout = dx_dropout
        self.char_dropout = char_dropout
        self.dx_threshold = 0
        self.char_threshold = 0
        self.attributions = []
        self.y_annotations = []
        self.y_char = []
        self.lossC_list = []
        self.lossA_list = []
        self.lossD_list = []
        self.dataset = []
        self.masks = []
        self.train_batch = []
        self.val_batch = []

        self.train_set, self.val_set, self.test_set, self.external_set = None, None, None, None
        
        self.lossD = nn.BCEWithLogitsLoss(pos_weight=dx_pos_weight)
        self.lossC = nn.BCEWithLogitsLoss(pos_weight=char_pos_weight, reduction='none')
        self.lossA = DiceLoss(reduction='none')
        
        self.labels = char_class_labels
        self.num_classes = len(char_class_labels)

        self.train_transform, self.test_transform = get_transforms(image_size=image_size)

        self.base_model = models.resnet50(pretrained=True)
        self.base_model.fc = nn.Identity()  # Remove the fully connected layer

        #self.diagnosis_head = nn.Linear(2048, 1)
        #self.characteristics_head = nn.Linear(2048, len(char_class_labels))
        
        self.diagnosis_head = nn.Sequential(
            nn.Dropout(self.dx_dropout),
            nn.Linear(2048, 1)
        )

        self.characteristics_head = nn.Sequential(
            nn.Dropout(self.char_dropout),
            nn.Linear(2048, len(char_class_labels))
        )

        self.layer_gc = LayerGradCam(self.base_model, self.base_model.layer4[-1])

        self.sigmoid = nn.Sigmoid()

        self.accuracy = Accuracy(threshold=0.5, average='macro', num_classes=self.num_classes)
        self.auroc = AUROC(average='macro', num_classes=self.num_classes)
        self.sensitivity = Recall(threshold=0.5, average='macro', num_classes=self.num_classes)
        self.specificity = Specificity(threshold=0.5, average='macro', num_classes=self.num_classes)
        
        self.target_layer = 'base_model.layer4'
        self.fmap = None
        self.grad = None
        self.handlers = []

        def save_fmaps(key):
            def hook(module, fmap_in, fmap_out):
                if isinstance(fmap_out, list):
                    fmap_out = fmap_out[-1]
                self.fmap = fmap_out

            return hook

        def save_grads(key):
            def hook(module, grad_in, grad_out):
                self.grad = grad_out[0]

            return hook

        target_module = dict(self.named_modules())[self.target_layer]
        self.handlers.append(
            target_module.register_forward_hook(save_fmaps(self.target_layer))
        )
        self.handlers.append(
            target_module.register_backward_hook(save_grads(self.target_layer))
        )

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
    
    def attribute(self, logits, class_idx):
        self.zero_grad()
        one_hot = F.one_hot((torch.ones(logits.size(0)) * class_idx).long(), self.num_classes).to(self.device)
        logits.backward(gradient=one_hot, retain_graph=True)
        weights = F.adaptive_avg_pool2d(self.grad, 1)
        attr = torch.mul(self.fmap, weights).sum(dim=1, keepdim=True)
        #attr = self.sigmoid(F.relu(attr)) * 2 - 1
        attr = F.relu(attr)
        return attr

    def forward(self, x, zero_non_predicted=True):
        
        output = self.base_model(x)
        diagnosis_output = self.diagnosis_head(output)
        characteristics_output = self.characteristics_head(output)
                
        # Get attributions for each class
        #attributions = torch.cat([self.layer_gc.attribute(x, class_idx, relu_attributions=True) for class_idx in range(self.num_classes)], dim=1)
        attributions = torch.cat([self.attribute(characteristics_output, class_idx) for class_idx in range(self.num_classes)], dim=1)
        # Upsample the attributions to match the image size
        attributions = F.interpolate(attributions, size=(image_size, image_size), mode='nearest')
        
        predicted_classes = torch.round(self.sigmoid(characteristics_output))
        # Set attributions of non predicted classes to zero by multiplying with the rounded prediction: 1 or 0.
        if zero_non_predicted:
            for batch_idx in range(attributions.size(0)):
                for class_idx in range(self.num_classes):
                    attributions[batch_idx][class_idx] *= predicted_classes[batch_idx][class_idx]
        
        return diagnosis_output, characteristics_output, attributions

    def on_train_start(self):

        self.log("hp/attention_weight", float(attention_weight))
        self.log("hp/lr", learning_rate)
        self.log("hp/batch_size", float(batch_size))
        self.log("hp/dropout", dx_dropout)
        self.log("hp/weighted_sampling", float(weighted_sampling))

    def training_step(self, batch, batch_idx):
        x, y = batch
                        
        y_dx, y_char, y_annotations, image_name = y
        diagnosis_predictions, characteristics_predictions, attributions = self(x)
        lossD = self.lossD(diagnosis_predictions, y_dx)    
        
        mask = torch.logical_not(torch.all(y_char == 0, dim=1)).int()
        lossC = torch.mean(self.lossC(characteristics_predictions, y_char).mean(1) * mask)
        lossA = torch.mean(self.lossA(attributions, y_annotations).mean() * mask)
            
        loss = lossD*dx_weight + lossC*char_weight + lossA*attention_weight        
        
        self.log("train/loss", loss, on_epoch=True, on_step=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        
        with torch.set_grad_enabled(True):
            diagnosis_predictions, characteristics_predictions, attributions = self(x)
        
        loss = self.lossD(diagnosis_predictions, y_dx)  
                
        self.log("val/loss", loss, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        
        with torch.set_grad_enabled(True):
            diagnosis_predictions, characteristics_predictions, attributions = self(x)
        
        loss = self.lossD(diagnosis_predictions, y_dx)    
        #loss = lossD * dx_weight + lossC * char_weight + lossA * attention_weight

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
        
        with torch.set_grad_enabled(True):
            diagnosis_predictions, characteristics_predictions, attributions = self(x)
        
        return diagnosis_predictions, characteristics_predictions, attributions, y_dx, y_char, image_name, self.inverse_normalize(x)
    
    def remove_hook(self):
        for handle in self.handlers:
            handle.remove()

    ####################
    # DATA RELATED HOOKS
    ####################

    def setup(self, stage=None):
        if stage == 'fit':
            metadata = pd.read_csv(metadata_file)#.drop_duplicates(subset='lesion_id', keep='last')
            
            train_set = metadata[metadata.split == 'train'] 
            val_set = metadata[metadata.split == 'val']
            test_set = metadata[metadata.split == 'test']
            external_set = metadata[metadata.split == 'external']
                        
            # Set positive class loss weight based on the distribution in the train set
            dist = train_set[dx_class_label].value_counts()
            pos_weight = [dist[0]/dist[1]]
            self.lossD.pos_weight = torch.Tensor(pos_weight)
            
            
            # Set class label loss weights based on their distributions in the train set
            pos_weight_dict = {}
            for label in char_class_labels:
                dist = train_set[label].value_counts()
                pos_weight_dict[label] = dist.loc[0]/dist.loc[1]
            pos_weight = [pos_weight_dict[label] for label in char_class_labels]
            self.lossC.pos_weight = torch.Tensor(pos_weight)
            
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


    
    
    
    
    
    
class SCPPretrain(pl.LightningModule):
    def __init__(self, hidden_size=64, learning_rate=2e-4, img_dir='./', annotations_dir='./', metadata_file='./', train_data_dir='./', val_data_dir='./',
                 test_data_dir='./', batch_size=32, dx_dropout=0.5, char_dropout=0.5, weighted_sampling=False, char_pos_weight=char_pos_weight,
                 dx_pos_weight=dx_pos_weight):

        super().__init__()
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.metadata_file = metadata_file
        
        self.weighted_sampling = weighted_sampling
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.dx_dropout = dx_dropout
        self.char_dropout = char_dropout
        self.attributions = []
        self.y_annotations = []
        self.y_char = []
        self.lossC_list = []
        self.lossA_list = []
        self.lossD_list = []
        self.dataset = []
        self.masks = []
        self.batches = []
        self.val_batch = []

        self.train_set, self.val_set, self.test_set = None, None, None
        
        self.lossD = nn.BCEWithLogitsLoss(pos_weight=dx_pos_weight)
        self.lossC = nn.BCEWithLogitsLoss(pos_weight=char_pos_weight)
        self.lossA = DiceLoss()
        
        self.labels = char_class_labels
        self.num_classes = len(char_class_labels)

        self.train_transform, self.test_transform = get_transforms(image_size=image_size)

        self.base_model = models.resnet50(pretrained=True)
        self.base_model.fc = nn.Identity()  # Remove the fully connected layer

        #self.diagnosis_head = nn.Linear(2048, 1)
        #self.characteristics_head = nn.Linear(2048, len(char_class_labels))
        
        self.diagnosis_head = nn.Sequential(
            nn.Dropout(self.dx_dropout),
            nn.Linear(2048, 1)
        )

        self.characteristics_head = nn.Sequential(
            nn.Dropout(self.char_dropout),
            nn.Linear(2048, len(char_class_labels))
        )

        self.layer_gc = LayerGradCam(self.base_model, self.base_model.layer4[-1])

        self.sigmoid = nn.Sigmoid()

        self.accuracy = Accuracy(threshold=0.5, average='macro', num_classes=self.num_classes)
        self.auroc = AUROC(average='macro', num_classes=self.num_classes)
        self.sensitivity = Recall(threshold=0.5, average='macro', num_classes=self.num_classes)
        self.specificity = Specificity(threshold=0.5, average='macro', num_classes=self.num_classes)
        
        self.target_layer = 'base_model.layer4'
        self.fmap = None
        self.grad = None
        self.handlers = []

        def save_fmaps(key):
            def hook(module, fmap_in, fmap_out):
                if isinstance(fmap_out, list):
                    fmap_out = fmap_out[-1]
                self.fmap = fmap_out

            return hook

        def save_grads(key):
            def hook(module, grad_in, grad_out):
                self.grad = grad_out[0]

            return hook

        target_module = dict(self.named_modules())[self.target_layer]
        self.handlers.append(
            target_module.register_forward_hook(save_fmaps(self.target_layer))
        )
        self.handlers.append(
            target_module.register_backward_hook(save_grads(self.target_layer))
        )

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
    
    def attribute(self, logits, class_idx):
        self.zero_grad()
        one_hot = F.one_hot((torch.ones(logits.size(0)) * class_idx).long(), self.num_classes).to(self.device)
        logits.backward(gradient=one_hot, retain_graph=True)
        weights = F.adaptive_avg_pool2d(self.grad, 1)
        attr = torch.mul(self.fmap, weights).sum(dim=1, keepdim=True)
        #attr = self.sigmoid(F.relu(attr)) * 2 - 1
        attr = F.relu(attr)
        return attr

    def forward(self, x, zero_non_predicted=True):
        
        output = self.base_model(x)
        diagnosis_output = self.diagnosis_head(output)
        characteristics_output = self.characteristics_head(output)
                
        # Get attributions for each class
        #attributions = torch.cat([self.layer_gc.attribute(x, class_idx, relu_attributions=True) for class_idx in range(self.num_classes)], dim=1)
        attributions = torch.cat([self.attribute(characteristics_output, class_idx) for class_idx in range(self.num_classes)], dim=1)
        # Upsample the attributions to match the image size
        attributions = F.interpolate(attributions, size=(image_size, image_size), mode='nearest')
        
        predicted_classes = torch.round(self.sigmoid(characteristics_output))
        # Set attributions of non predicted classes to zero by multiplying with the rounded prediction: 1 or 0.
        if zero_non_predicted:
            for batch_idx in range(attributions.size(0)):
                for class_idx in range(self.num_classes):
                    attributions[batch_idx][class_idx] *= predicted_classes[batch_idx][class_idx]
        
        return diagnosis_output, characteristics_output, attributions

    def on_train_start(self):

        self.log("hp/attention_weight", float(attention_weight))
        self.log("hp/lr", learning_rate)
        self.log("hp/batch_size", float(batch_size))
        self.log("hp/dropout", dx_dropout)
        self.log("hp/weighted_sampling", float(weighted_sampling))
        
    def split_batch(self, batch):
        x, y = batch
        char_idx = np.array([True if item.sum() > 0 else False for item in y[1]])
        
        x1 = x[~char_idx]
        x2 = x[char_idx]
        y1 = [y[i][~char_idx] for i in range(3)]
        y2 = [y[i][char_idx] for i in range(3)]
        y1.append(tuple(value for value, condition in zip(y[3], char_idx) if not condition))
        y2.append(tuple(value for value, condition in zip(y[3], char_idx) if condition))
        return x1, y1, x2, y2

    def training_step(self, batch, batch_idx):
        x, y = batch
                        
        y_dx, y_char, y_annotations, image_name = y
        diagnosis_predictions, characteristics_predictions, attributions = self(x)
        lossD = self.lossD(diagnosis_predictions, y_dx)    
        
        mask = torch.logical_not(torch.all(y_char == 0, dim=1)).int()
        lossC = torch.mean(self.lossC(characteristics_predictions, y_char).mean(1) * mask)
        lossA = torch.mean(self.lossA(attributions, y_annotations).mean() * mask)
            
        loss = lossD*dx_weight + lossC*char_weight + lossA*attention_weight        
        
        self.log("train/loss", loss, on_epoch=True, on_step=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        
        with torch.set_grad_enabled(True):
            diagnosis_predictions, characteristics_predictions, attributions = self(x)
        
        loss = self.lossD(diagnosis_predictions, y_dx)  
        
        self.log("val/loss", loss, on_epoch=True, on_step=False)
        #self.log("val/lossA", lossA, on_epoch=True, on_step=False)
        #self.log("val/lossC", lossC, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        
        with torch.set_grad_enabled(True):
            diagnosis_predictions, characteristics_predictions, attributions = self(x)
        
        loss = self.lossD(diagnosis_predictions, y_dx)    
        #loss = lossD * dx_weight + lossC * char_weight + lossA * attention_weight

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
        
        with torch.set_grad_enabled(True):
            diagnosis_predictions, characteristics_predictions, attributions = self(x)
        
        return diagnosis_predictions, characteristics_predictions, attributions, y_dx, y_char, image_name, self.inverse_normalize(x)
    
    def remove_hook(self):
        for handle in self.handlers:
            handle.remove()

    ####################
    # DATA RELATED HOOKS
    ####################

    def setup(self, stage=None):
        if stage == 'fit':
            metadata = pd.read_csv(metadata_file)#.drop_duplicates(subset='lesion_id', keep='last')
            metadata = metadata[metadata.dataset == 'scp']
            
            train_set, test_set = train_test_split(metadata, test_size=0.18, random_state=42)
            train_set, val_set = train_test_split(train_set, test_size=0.18, random_state=42)
            
            
            # Set positive class loss weight based on the distribution in the train set
            dist = train_set[dx_class_label].value_counts()
            pos_weight = [dist[0]/dist[1]]
            self.lossD.pos_weight = torch.Tensor(pos_weight)
            
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
    

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)



    
    
    
    
    

class HAMFineTune(pl.LightningModule):
    def __init__(self, hidden_size=64, learning_rate=2e-4, img_dir='./', annotations_dir='./', metadata_file='./', train_data_dir='./',
                 val_data_dir='./', test_data_dir='./', batch_size=32, dx_dropout=0.5, char_dropout=0.5, weighted_sampling=False,
                 char_pos_weight=char_pos_weight, dx_pos_weight=dx_pos_weight):

        super().__init__()
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.metadata_file = metadata_file
        
        self.weighted_sampling = weighted_sampling
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.dx_dropout = dx_dropout
        self.char_dropout = char_dropout
        self.attributions = []
        self.y_annotations = []
        self.y_char = []
        self.lossC_list = []
        self.lossA_list = []
        self.lossD_list = []
        self.dataset = []
        self.masks = []
        self.batches = []
        self.val_batch = []

        self.train_set, self.val_set, self.test_set, self.external_set = None, None, None, None
        
        self.lossD = nn.BCEWithLogitsLoss(pos_weight=dx_pos_weight)
        self.lossC = nn.BCEWithLogitsLoss(pos_weight=char_pos_weight, reduction='none')
        self.lossA = DiceLoss(reduction='none')
        
        self.labels = char_class_labels
        self.num_classes = len(char_class_labels)

        self.train_transform, self.test_transform = get_transforms(image_size=image_size)

        self.base_model = models.resnet50(pretrained=True)
        self.base_model.fc = nn.Identity()  # Remove the fully connected layer

        #self.diagnosis_head = nn.Linear(2048, 1)
        #self.characteristics_head = nn.Linear(2048, len(char_class_labels))
        
        self.diagnosis_head = nn.Sequential(
            nn.Dropout(self.dx_dropout),
            nn.Linear(2048, 1)
        )

        self.characteristics_head = nn.Sequential(
            nn.Dropout(self.char_dropout),
            nn.Linear(2048, len(char_class_labels))
        )

        self.layer_gc = LayerGradCam(self.base_model, self.base_model.layer4[-1])

        self.sigmoid = nn.Sigmoid()

        self.accuracy = Accuracy(threshold=0.5, average='macro', num_classes=self.num_classes)
        self.auroc = AUROC(average='macro', num_classes=self.num_classes)
        self.sensitivity = Recall(threshold=0.5, average='macro', num_classes=self.num_classes)
        self.specificity = Specificity(threshold=0.5, average='macro', num_classes=self.num_classes)
        
        self.target_layer = 'base_model.layer4'
        self.fmap = None
        self.grad = None
        self.handlers = []

        def save_fmaps(key):
            def hook(module, fmap_in, fmap_out):
                if isinstance(fmap_out, list):
                    fmap_out = fmap_out[-1]
                self.fmap = fmap_out

            return hook

        def save_grads(key):
            def hook(module, grad_in, grad_out):
                self.grad = grad_out[0]

            return hook

        target_module = dict(self.named_modules())[self.target_layer]
        self.handlers.append(
            target_module.register_forward_hook(save_fmaps(self.target_layer))
        )
        self.handlers.append(
            target_module.register_backward_hook(save_grads(self.target_layer))
        )

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
    
    def attribute(self, logits, class_idx):
        self.zero_grad()
        one_hot = F.one_hot((torch.ones(logits.size(0)) * class_idx).long(), self.num_classes).to(self.device)
        logits.backward(gradient=one_hot, retain_graph=True)
        weights = F.adaptive_avg_pool2d(self.grad, 1)
        attr = torch.mul(self.fmap, weights).sum(dim=1, keepdim=True)
        #attr = self.sigmoid(F.relu(attr)) * 2 - 1
        attr = F.relu(attr)
        return attr

    def forward(self, x, zero_non_predicted=True):
        
        output = self.base_model(x)
        diagnosis_output = self.diagnosis_head(output)
        characteristics_output = self.characteristics_head(output)
                
        # Get attributions for each class
        #attributions = torch.cat([self.layer_gc.attribute(x, class_idx, relu_attributions=True) for class_idx in range(self.num_classes)], dim=1)
        attributions = torch.cat([self.attribute(characteristics_output, class_idx) for class_idx in range(self.num_classes)], dim=1)
        # Upsample the attributions to match the image size
        attributions = F.interpolate(attributions, size=(image_size, image_size), mode='nearest')
        
        predicted_classes = torch.round(self.sigmoid(characteristics_output))
        # Set attributions of non predicted classes to zero by multiplying with the rounded prediction: 1 or 0.
        if zero_non_predicted:
            for batch_idx in range(attributions.size(0)):
                for class_idx in range(self.num_classes):
                    attributions[batch_idx][class_idx] *= predicted_classes[batch_idx][class_idx]
        
        return diagnosis_output, characteristics_output, attributions

    def on_train_start(self):

        self.log("hp/attention_weight", float(attention_weight))
        self.log("hp/lr", learning_rate)
        self.log("hp/batch_size", float(batch_size))
        self.log("hp/dropout", dx_dropout)
        self.log("hp/weighted_sampling", float(weighted_sampling))

    def training_step(self, batch, batch_idx):
        x, y = batch
                        
        y_dx, y_char, y_annotations, image_name = y
        diagnosis_predictions, characteristics_predictions, attributions = self(x)
        lossD = self.lossD(diagnosis_predictions, y_dx)    
        
        mask = torch.logical_not(torch.all(y_char == 0, dim=1)).int()
        lossC = torch.mean(self.lossC(characteristics_predictions, y_char).mean(1) * mask)
        lossA = torch.mean(self.lossA(attributions, y_annotations).mean() * mask)
            
        loss = lossD*dx_weight + lossC*char_weight + lossA*attention_weight        
        
        self.log("train/loss", loss, on_epoch=True, on_step=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        
        with torch.set_grad_enabled(True):
            diagnosis_predictions, characteristics_predictions, attributions = self(x)
        
        loss = self.lossD(diagnosis_predictions, y_dx)  
        
        self.log("val/loss", loss, on_epoch=True, on_step=False)
        #self.log("val/lossA", lossA, on_epoch=True, on_step=False)
        #self.log("val/lossC", lossC, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        
        with torch.set_grad_enabled(True):
            diagnosis_predictions, characteristics_predictions, attributions = self(x)
        
        loss = self.lossD(diagnosis_predictions, y_dx)    
        #loss = lossD * dx_weight + lossC * char_weight + lossA * attention_weight

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
        
        with torch.set_grad_enabled(True):
            diagnosis_predictions, characteristics_predictions, attributions = self(x)
        
        return diagnosis_predictions, characteristics_predictions, attributions, y_dx, y_char, image_name, self.inverse_normalize(x)
    
    def remove_hook(self):
        for handle in self.handlers:
            handle.remove()

    ####################
    # DATA RELATED HOOKS
    ####################

    def setup(self, stage=None):
        if stage == 'fit':
            metadata = pd.read_csv(metadata_file).drop_duplicates(subset='lesion_id', keep='last')
            
            train_set = metadata[metadata.split == 'train'] 
            val_set = metadata[metadata.split == 'val']
            test_set = metadata[metadata.split == 'test']
            external_set = metadata[metadata.split == 'external']
            
            train_set = train_set[train_set.dataset.isin(['ham'])]
            #val_set = val_set[val_set.dataset.isin(['ham'])]
            
            # Set positive class loss weight based on the distribution in the train set
            dist = train_set[dx_class_label].value_counts()
            pos_weight = [dist[0]/dist[1]]
            self.lossD.pos_weight = torch.Tensor(pos_weight)
            
            
            # Set class label loss weights based on their distributions in the train set
            pos_weight_dict = {}
            for label in char_class_labels:
                dist = train_set[label].value_counts()
                pos_weight_dict[label] = dist.loc[0]/dist.loc[1]
            pos_weight = [pos_weight_dict[label] for label in char_class_labels]
            self.lossC.pos_weight = torch.Tensor(pos_weight)
            
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


    