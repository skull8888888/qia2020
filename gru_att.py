
import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pytorch_lightning.metrics.functional as plm

import torch.nn as nn
import torchvision.models as models

from pytorch_lightning.metrics.functional import accuracy

import pytorch_lightning as pl

from models.inception_resnet_v1 import InceptionResnetV1

from torch.optim.lr_scheduler import StepLR

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        
        hidden_size = 200
        
        self.alpha = nn.Linear(hidden_size, 50)
        self.drop_p = 0.5
        self.pred_fc1 = nn.Linear(hidden_size, 7)
        
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        
    def attention(self, embedded, encoder_outputs):

        t = embedded.size(1)
        
        alphas = F.softmax(self.alpha(embedded), dim=1)
        attn_applied = torch.bmm(alphas, encoder_outputs)

        output = attn_applied.sum(1)

        return output
        
        
    def forward(self, inp):
        
        encoder_outputs, hidden = self.gru(inp, None)
        
        attn = self.attention(inp, encoder_outputs)
        attn = F.dropout(attn, p=self.drop_p, training=self.training)
        pred_score = self.pred_fc1(attn)

        return pred_score
    

class GRU_ATT(pl.LightningModule):
    
    def __init__(self):
        super(GRU_ATT, self).__init__()

        self.decoder = Attention()
            
#          {'hap':0, 'sur':1, 'neu':2, 'fea':3, 'dis':4, 'ang':5, 'sad':6}
        self.register_buffer("w", torch.Tensor([1.,1.,0.5,1.,1.,1.,0.75]))
        
            
    def forward(self, x):
        return self.decoder(x)
    
    def training_step(self, batch, batch_nb):
        X, y = batch
        y_hat = self.forward(X)
       
        y = y.squeeze_()
           
        loss = F.cross_entropy(y_hat, y, weight=self.w)
        self.log('train_loss', loss)
        return loss
    
    
    def validation_step(self, batch, batch_nb):
        X, y = batch
       
        y_hat = self.forward(X)
        y_pred = y_hat.max(1)[1]
        
        y = y.squeeze_()
        
        loss = F.cross_entropy(y_hat, y, weight=self.w)
      
        return {'val_loss': loss, 'correct_count': (y_pred == y).sum(), 'all_count': y.size(0)}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        
        correct_count = 0
        all_count = 0
        
        for l in outputs:
            correct_count += l['correct_count']
            all_count += l['all_count']
        
        self.log('val_loss', avg_loss,prog_bar=True)
        self.log('val_acc', correct_count / all_count, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
        return [optimizer], [scheduler]
#         optimizer = torch.optim.SGD(self.parameters(), lr=4e-6, momentum=0.9, weight_decay=1e-4)
        return optimizer
                                                
                                                
                                               
                                                
    
