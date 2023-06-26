import pandas as pd
import data_loader
import scipy
import numpy as np
import sys
import torch
import torch.nn as nn
from EcgDataset import EcgDataset
from EcgRecognitionModel import EcgRecognitionModel
import time

def read_diagnosis():
    diag_frame = pd.read_csv('diagnoses.csv')
    diag_dict = {}
    for line in diag_frame.values:
        diag_dict[line[0]] = line[1]
    return diag_dict

train_data, train_labels = data_loader.load_dataset('./train.mat')
train_dataset = EcgDataset(train_data, train_labels)
#print(len(train_dataset[0][0]))


#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#print(train_dataset[0][0].shape)
device = "cuda"

t2 = train_dataset.ecgs.to(device)
model = EcgRecognitionModel().to(device)

print(model(t2[0]))
