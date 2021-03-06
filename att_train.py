import torch
import torchvision
from torch.utils import data

import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything

from os import path

from functions import *
from gru_att_new import GRU_ATT
import pandas as pd

df = pd.read_csv('videoname_map.csv')
y_df = pd.read_csv('clean.csv')

val_df = pd.read_csv('val_videoname_map.csv')
val_y_df = pd.read_csv('qia2020/val.csv')


class TorchVideoTrainDataset(data.Dataset):
    
    def __init__(self, path, X_df, y_df):

        self.path = path
        self.X_df = X_df
        self.y_df = y_df
   
    def __len__(self):
        
        return len(self.y_df)
    
    def __getitem__(self, index):
        "Generates one sample of data"
#         print(index)
        emo2index = {'hap':0, 'sur':1, 'neu':2, 'fea':3, 'dis':4, 'ang':5, 'sad':6}
        
        filename = str(self.y_df.FileID.iloc[index]).zfill(5)
        
        with np.load(self.path + filename +'.npz') as data:
            T = torch.Tensor(data['word_embed'])
        
        if T.size(0) < 50:
            T = torch.cat([T,torch.zeros(50-T.size(0),200)])
        
        i = self.y_df.Emotion.iloc[index]

        y = torch.LongTensor([emo2index[i]])
        return T, y
        
batch_size = 128

video_dataset = TorchVideoTrainDataset('qia2020/train/', df, y_df)
train_loader = DataLoader(video_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)

val_video_dataset = TorchVideoTrainDataset('qia2020/val/', val_df, val_y_df)
val_loader = DataLoader(val_video_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8, pin_memory=True)


# checkpoint_dir = 'att/lightning_logs/version_8/checkpoints/epoch=99.ckpt'

system = GRU_ATT()

seed_everything(42)


# trainer = Trainer(gpus=[1], accelerator='ddp', resume_from_checkpoint=checkpoint_dir, deterministic=True, max_epochs=200, default_root_dir='/home/user/robert/att')
trainer = Trainer(gpus=[1], max_epochs=100, default_root_dir='/home/user/robert/att')
trainer.fit(system, train_loader, val_loader)



