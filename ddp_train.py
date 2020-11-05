#!/usr/bin/env python
# coding: utf-8

import torch
import torchvision
from torch.utils import data
import cv2
from PIL import Image
from models.mtcnn import MTCNN

import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything

from os import path

from functions import *
from crnn import CRNN
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('videoname_map.csv')
y_df = pd.read_csv('qia2020/train.csv')

val_df = pd.read_csv('val_videoname_map.csv')
val_y_df = pd.read_csv('qia2020/val.csv')

# df = pd.read_csv('videoname_map.csv')
# y_df = pd.read_csv('qia2020/train.csv')

# val_df = pd.read_csv('val_videoname_map.csv')
# val_y_df = pd.read_csv('qia2020/val.csv')

class TorchVideoTrainDataset(data.Dataset):
    
    def __init__(self, torch_path, np_path, X_df, y_df, l):

        self.torch_path = torch_path
        self.np_path = np_path
        self.X_df = X_df
        self.y_df = y_df
        self.l = l
    
    def __len__(self):
        
        return self.l
    
    def __getitem__(self, index):
        
#         print(index)
        emo2index = {'hap':0, 'sur':1, 'neu':2, 'fea':3, 'dis':4, 'ang':5, 'sad':6}
        
        filename = str(self.y_df.FileID.iloc[index]).zfill(5)
        
        with np.load(self.np_path + filename +'.npz') as data:
            T = torch.Tensor(data['word_embed'])
        
        if T.size(0) < 50:
            T = torch.cat([T,torch.zeros(50-T.size(0),200)])
        
        X = torch.load(self.torch_path + filename + '.pt')
        
        y = torch.LongTensor([emo2index[self.y_df.Emotion.iloc[index]]])
        return (X, T) , y


batch_size = 90

video_dataset = TorchVideoTrainDataset('torch_video_3/', 'qia2020/train/', df, y_df, 40000)
train_loader = DataLoader(video_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)

val_video_dataset = TorchVideoTrainDataset('torch_video_3_val/', 'qia2020/val/', val_df, val_y_df, 5000)
val_loader = DataLoader(val_video_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8, pin_memory=True)


# checkpoint_dir = 'lightning_logs/version_39/checkpoints/epoch=1.ckpt'
system = CRNN()

seed_everything(42)

# trainer = Trainer(gpus=[0], accelerator='ddp', resume_from_checkpoint=checkpoint_dir, deterministic=True, max_epochs=100)
trainer = Trainer(gpus=[0], max_epochs=100, deterministic=True)
trainer.fit(system, train_loader, val_loader)



