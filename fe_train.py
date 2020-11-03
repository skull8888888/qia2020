#!/usr/bin/env python
# coding: utf-8

import torch
import torchvision
from torch.utils import data

import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything

# from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt

from os import path

from functions import *
from fe import FE
import pandas as pd
import random

df = pd.read_csv('videoname_map.csv')
y_df = pd.read_csv('qia2020/train_face.csv')

val_df = pd.read_csv('val_videoname_map.csv')
val_y_df = pd.read_csv('qia2020/val_face.csv')

class TorchVideoTrainDataset(data.Dataset):
    
    def __init__(self, path, X_df, y_df, l):

        self.path = path
#         ../../../emotion/qia2020/train/
        self.X_df = X_df
        self.y_df = y_df
        self.l = l
    
    def __len__(self):
        
        return self.l
    
    def __getitem__(self, index):
        "Generates one sample of data"
#         print(index)
        emo2index = {'hap':0, 'sur':1, 'neu':2, 'fea':3, 'dis':4, 'ang':5, 'sad':6}
        
        X = torch.load(self.path + str(self.y_df.FileID[index]).zfill(5) + '.pt')
        
        if X.size(0) > 3:
            X = X[:3,:,:,:]
        
        i = random.randint(0, 2)
        X = X[i].squeeze(0)
        
        y = torch.LongTensor([emo2index[self.y_df.Emotion[index]]])
        return X, y

batch_size = 64

video_dataset = TorchVideoTrainDataset('torch_video_3/', df, y_df, 40000)
train_loader = DataLoader(video_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)

val_video_dataset = TorchVideoTrainDataset('torch_video_3_val/', val_df, val_y_df, 5000)
val_loader = DataLoader(val_video_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8, pin_memory=True)

# val_video_dataset = VideoTrainDataset('../../../emotion/qia2020/val/', 'torch_video_val/', val_df, val_y_df)
# val_loader = DataLoader(val_video_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8, pin_memory=True)


# checkpoint_dir = 'fe/lightning_logs/version_2/checkpoints/epoch=1.ckpt'
system = FE()

seed_everything(42)


# trainer = Trainer(gpus=[2,3], accelerator='ddp', resume_from_checkpoint=checkpoint_dir, deterministic=True, max_epochs=100, default_root_dir='/mnt/home/20190718/robert/fe')
trainer = Trainer(gpus=[0,1], accelerator='ddp', max_epochs=100, deterministic=True, default_root_dir='/home/user/robert/fe')
trainer.fit(system, train_loader, val_loader)



