
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

class ResCNNEncoder(nn.Module):
    def __init__(self):
        super(ResCNNEncoder, self).__init__()

#         self.resnet = InceptionResnetV1(pretrained='vggface2')           

        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, 512)
        
        self.linear = nn.Linear(512,7)

    def forward(self, x):
        x = self.resnet(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear(x)
        return x

class FE(pl.LightningModule):
    
    def __init__(self):
        super(FE, self).__init__()
        
        self.resnet = ResCNNEncoder()
#          {'hap':0, 'sur':1, 'neu':2, 'fea':3, 'dis':4, 'ang':5, 'sad':6}
        self.register_buffer("w", torch.Tensor([1.,1.,0.5,1.,1.,1.,1.]))

                    
    def forward(self, x):
        return self.resnet(x)
    
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
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-6, momentum=0.9, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        return [optimizer], [scheduler]
    
#         optimizer = torch.optim.SGD(self.parameters(), lr=5e-4, momentum=0.9)
#         return optimizer
    


# import os
# import torch
# from torch.nn import functional as F
# from torch.utils.data import DataLoader
# import torchvision.transforms as transforms
# import pytorch_lightning.metrics.functional as plm

# import torch.nn as nn
# import torchvision.models as models

# from pytorch_lightning.metrics.functional import accuracy

# import pytorch_lightning as pl

# from models.inception_resnet_v1 import InceptionResnetV1

# from torch.optim.lr_scheduler import StepLR

# class ResCNNEncoder(nn.Module):
#     def __init__(self):
#         super(ResCNNEncoder, self).__init__()

# #         self.resnet = InceptionResnetV1(pretrained='vggface2')           

#         self.resnet = models.resnet18(pretrained=True)
#         self.resnet.fc = nn.Linear(512, 512)
        
#         self.linear = nn.Linear(512,7)

#     def forward(self, x):
#         x = self.resnet(x)
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.linear(x)
#         return x

# class FE(pl.LightningModule):
    
#     def __init__(self):
#         super(FE, self).__init__()
        
#         self.resnet = ResCNNEncoder()
# #          {'hap':0, 'sur':1, 'neu':2, 'fea':3, 'dis':4, 'ang':5, 'sad':6}
#         self.register_buffer("w", torch.Tensor([1.,1.,0.5,1.,1.,1.,1.]))

                    
#     def forward(self, x):
#         return self.resnet(x)
    
#     def training_step(self, batch, batch_nb):
#         X, y = batch
#         y_hat = self.forward(X)
                  
#         loss = F.binary_cross_entropy_with_logits(y_hat, y, weight=self.w)
#         self.log('train_loss', loss)
#         return loss
    
    
#     def validation_step(self, batch, batch_nb):
#         X, y = batch
       
#         y_hat = self.forward(X)
#         y_pred = (torch.sigmoid(y_hat) > 0.5).to(torch.int)
        
#         nc = (abs(y_pred - y.to(torch.int)).sum(dim=1) > 0).sum()
            
#         loss = F.binary_cross_entropy_with_logits(y_hat, y, weight=self.w)
      
#         return {'val_loss': loss, 'correct_count': y.size(0) - nc, 'all_count': y.size(0)}
    
#     def validation_epoch_end(self, outputs):
#         avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        
#         correct_count = 0
#         all_count = 0
        
#         for l in outputs:
#             correct_count += l['correct_count']
#             all_count += l['all_count']
        
#         self.log('val_loss', avg_loss,prog_bar=True)
#         self.log('val_acc', correct_count / all_count, prog_bar=True)
    
#     def configure_optimizers(self):
#         optimizer = torch.optim.SGD(self.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
#         scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
#         return [optimizer], [scheduler]
    
# #         optimizer = torch.optim.SGD(self.parameters(), lr=5e-4, momentum=0.9)
# #         return optimizer
    
