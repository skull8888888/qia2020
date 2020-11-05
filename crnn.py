
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
from fe import ResCNNEncoder as FE

class ResCNNEncoder(nn.Module):
    def __init__(self, pretrained=True):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResCNNEncoder, self).__init__()
        
        self.resnet = FE()
        
        if pretrained:
            self.resnet.load_state_dict(torch.load('40k_3_fe.pt'))
        modules = list(self.resnet.children())[:-1] 
        self.resnet = nn.Sequential(*modules)
            
    def forward(self, x_3d):
        cnn_embed_seq = []
        
        for t in range(x_3d.size(1)):
            # ResNet CNN
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  # ResNet
                
            cnn_embed_seq.append(x)

        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=1)

        return cnn_embed_seq

class Attention(nn.Module):
    def __init__(self, CNN_embed_dim=512, drop_p=0.5, num_classes=7, pretrained=True):
        super(Attention, self).__init__()
        
        hidden_size = 200
        
        self.video_alpha = nn.Sequential(nn.Linear(CNN_embed_dim, 1),
                                   nn.Sigmoid())
        
        self.drop_p = drop_p
        self.pred_fc1 = nn.Linear(712, num_classes)
        
        self.text_alpha = nn.Linear(hidden_size, 50)
        
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        
        if pretrained:
            self.gru.load_state_dict(torch.load('40k_gru.pt'))
            self.text_alpha.load_state_dict(torch.load('40k_ta.pt'))

    def attention(self, x, alpha):

        t = x.size(1)
        
        alphas = []
        vs = []

        for i in range(t):
            f = x[:,i,:]
            vs.append(f)

            f = F.dropout(f, p=self.drop_p, training=self.training)

            a = alpha(f)
            alphas.append(a)
        
        
        vs_stack = torch.stack(vs, dim=2) # [B,512,t]
        alphas_stack = torch.stack(alphas, dim=2) #[B,1,t]
        
        vm1 = vs_stack.mul(alphas_stack).sum(2).div(alphas_stack.sum(2))
        
        return vm1
        
    def text_attention(self, embedded, encoder_outputs):

        t = embedded.size(1)
        
        alphas = F.softmax(self.text_alpha(embedded), dim=1)
        attn_applied = torch.bmm(alphas, encoder_outputs)

        output = attn_applied.sum(1)

        return output
        
    def forward(self, data):
        
        video = self.attention(data[0], self.video_alpha)
        
        #################### text
        encoder_outputs, hidden = self.gru(data[1], None)
        
        attn = self.text_attention(data[1], encoder_outputs)
        text = F.dropout(attn, p=self.drop_p, training=self.training)
        
        vm = torch.cat([video, text], dim=1) 
        vm = F.dropout(vm, p=self.drop_p, training=self.training)
       
        pred_score = self.pred_fc1(vm)

        return pred_score

class CRNN(pl.LightningModule):
    
    def __init__(self, pretrained=True):
        super(CRNN, self).__init__()

        self.encoder = ResCNNEncoder(pretrained=pretrained)
        self.decoder = Attention(pretrained=pretrained)
            
#          {'hap':0, 'sur':1, 'neu':2, 'fea':3, 'dis':4, 'ang':5, 'sad':6}
        self.register_buffer("w", torch.Tensor([1.,1.,0.8,1.,1.,1.,1.]))
        
            
    def forward(self, x):
        v = self.encoder(x[0])
        return self.decoder((v,x[1]))
    
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
        optimizer = torch.optim.SGD(self.parameters(), lr=5e-5, momentum=0.9, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        return [optimizer], [scheduler]
                                                
                                                
                                               
                                                
    
