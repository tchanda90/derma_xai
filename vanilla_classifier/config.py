import random
import os
import numpy as np
import pandas as pd
import torch


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

#img_dir = "/home/caduser/Tirtha/overlap/data/HAM10000/HAM10000"
img_dir = "/home/caduser/Tirtha/data/combined"
#annotations_dir = "/home/caduser/Tirtha/overlap/data/ground_truth/annotations_gt"
annotations_dir = "/home/caduser/Tirtha/data/HAM10000/annotations_gt"
metadata_file = "/home/caduser/Tirtha/data/metadata_gt_extended.csv"
#metadata_file = "metadata_undersampled.csv"


model_save_dir = "./models"

weighted_sampling = False
save_attention_plots = False

#params = {'batch_size': 16, 'dropout': 0.1, 'learning_rate': 0.00001}
params = {'batch_size': 32, 'dx_dropout': 0.5, 'char_dropout': 0.5, 'learning_rate': 0.0001} 
num_epochs = 25
learning_rate = params['learning_rate']
batch_size = params['batch_size']
dx_dropout = params['dx_dropout']
char_dropout = params['char_dropout']

image_size = 224
attention_weight = 1
char_weight = 2
dx_weight = 1

 
mel_class_labels = ['TRBL', 'ESA', 'BDG', 'GP', 'PV', 'PRL', 'WLSA', 'PLR', 'PES', 'PIF']
nev_class_labels = ['OPC', 'SPC', 'MVP', 'PRLC', 'PLF', 'PDES', 'APC', 'MS']
char_class_labels = mel_class_labels+nev_class_labels 
char_pos_weight = torch.tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1])

mel_class_labels = ['TRBL', 'BDG', 'GP', 'WLSA', 'PRL', 'PV']
nev_class_labels = ['SPC', 'APC', 'OPC', 'MVP']
char_class_labels = mel_class_labels+nev_class_labels 


dx_pos_weight = torch.tensor([1])
char_pos_weight = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])


char_class_labels_pred = [label+'_pred' for label in char_class_labels]
mel_class_labels_pred = [label+'_pred' for label in mel_class_labels]
nev_class_labels_pred = [label+'_pred' for label in nev_class_labels]
char_class_labels_score = [label+'_score' for label in char_class_labels]

annotation_labels = [label+'_annotation' for label in char_class_labels]

dx_class_label = ['benign_malignant']

seed = 42

seed_everything(seed)

char_full_mapping = {'TRBL': 'thick reticular or branched lines', 'BDG': 'black dots or globules in the periphery of the lesion', 
                    'WLSA': 'white lines or white structureless area', 'ESA': 'eccentrically located structureless area', 'GP': 'grey patterns',
                    'PV': 'polymorphous vessels', 'PRL': 'pseudopods or radial lines at the lesion margin that do not occupy the entire lesional circumference',
                    'APC': 'asymmetric combination of multiple patterns or colours in the absence of other melanoma criteria', 'MS': 'melanoma simulator', 
                    'OPC': 'only one pattern and only one colour', 'SPC': 'symmetric combination of patterns and colors', 
                    'MVP': 'monomorphic vascular patterns'}