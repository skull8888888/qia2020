
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

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x
class normalize(nn.Module):
    def __init__(self):
        super(normalize, self).__init__()
        
    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        return x

# class ResCNNEncoder(nn.Module):
#     def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
#         """Load the pretrained ResNet-152 and replace top fc layer."""
#         super(ResCNNEncoder, self).__init__()

#         self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
#         self.drop_p = drop_p

# #         resnet = models.resnet152(pretrained=True)
# #         modules = list(resnet.children())[:-1]      # delete the last fc layer.
# #         self.resnet = nn.Sequential(*modules)
# #         for p in self.resnet.parameters():
# #             p.requires_grad = False

#         self.resnet = InceptionResnetV1(pretrained='vggface2')
        
    
# #         self.resnet = nn.Sequential(*list(IR.children())[:-5])

#         for param in self.resnet.parameters():
#             param.requires_grad = False
            
# #         self.resnet.avgpool_1a = nn.AdaptiveAvgPool2d(output_size=1)
# #         self.resnet.last_linear = nn.Sequential(
# #             Flatten(),
# #             nn.Linear(in_features=1792, out_features=512, bias=False),
# #             normalize()
# #         )
        
# #         self.fc1 = nn.Linear(512, fc_hidden1)
# #         self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
# #         self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
# #         self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
# #         self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)
        
     
#     def forward(self, x_3d):
#         cnn_embed_seq = []
        
#         for t in range(x_3d.size(1)):
#             # ResNet CNN
#             with torch.no_grad():
#                 x = self.resnet(x_3d[:, t, :, :, :])  # ResNet
#                 x = x.view(x.size(0), -1)             # flatten output of conv

#             # FC layers
# #             x = self.bn1(self.fc1(x))
# #             x = F.relu(x)
# #             x = self.bn2(self.fc2(x))
# #             x = F.relu(x)
# #             x = F.dropout(x, p=self.drop_p, training=self.training)
# #             x = self.fc3(x)

#             cnn_embed_seq.append(x)

#         # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
#         cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
#         # cnn_embed_seq: shape=(batch, time_step, input_size)

#         return cnn_embed_seq

class ResCNNEncoder(nn.Module):
    def __init__(self):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResCNNEncoder, self).__init__()

#         self.resnet = InceptionResnetV1(pretrained='vggface2')
        
#         for param in self.resnet.parameters():
#             param.requires_grad = False
        
        self.resnet = FE()
#         self.resnet.load_state_dict(torch.load('40k_3_fe.pt'))
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
    def __init__(self, CNN_embed_dim=512, drop_p=0.5, num_classes=7):
        super(Attention, self).__init__()
        
        hidden_size = 200
        
        self.video_alpha = nn.Sequential(nn.Linear(CNN_embed_dim, 1),
                                   nn.Sigmoid())
        
        self.drop_p = drop_p
        self.pred_fc1 = nn.Linear(712, num_classes)
        
        self.text_alpha = nn.Linear(hidden_size, 50)
#         self.text_alpha.load_state_dict(torch.load('40k_ta.pt'))
        
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
#         self.gru.load_state_dict(torch.load('40k_gru.pt'))
        
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
    
    def __init__(self):
        super(CRNN, self).__init__()

        self.encoder = ResCNNEncoder()
        self.decoder = Attention()
            
#          {'hap':0, 'sur':1, 'neu':2, 'fea':3, 'dis':4, 'ang':5, 'sad':6}
        self.register_buffer("w", torch.Tensor([1.,1.,0.5,1.,1.,1.,1.]))
        
            
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
#         optimizer = torch.optim.SGD(self.parameters(), lr=4e-6, momentum=0.9, weight_decay=1e-4)
#         return optimizer
                                                
                                                
                                               
                                                
    
