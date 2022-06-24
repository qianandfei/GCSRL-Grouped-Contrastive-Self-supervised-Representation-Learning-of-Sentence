import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from models import TextCNN
from configs import get_args
from utils import data_selfattention
from torch.nn import CrossEntropyLoss
from transformers import Trainer


def cl_cal_loss(vec1,vec2):
    loss_for_cl=CrossEntropyLoss(ignore_index=-100)
    labels=torch.arange(0,vec1.shape[0],device='cuda')
    vec1=F.normalize(vec1, p=2, dim=1)#归一化为单位向量[bs,hiden_len]
    vec2=F.normalize(vec2, p=2, dim=1)#[bs,hiden_len]
    sims=vec1.matmul(vec2.T)*20
    loss=loss_for_cl(sims,labels)#拉近二者距离
    return loss


def cos_similarity(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception



class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=4096):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=4096, hidden_dim=1024, out_dim=4096): # bottleneck structure
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class Plus_Proj_layer(nn.Module):#   继承类nn.Module
    def __init__(self, backbone):
        super().__init__()
        model_opt = TextCNN.ModelConfig()
        args=get_args()
        self.backbone = backbone #

        if args.backbone=='textcnn':
            self.projector = projection_MLP(model_opt.model_dim,4096)
            
        else:
            self.projector = projection_MLP(768)
        self.encoder = nn.Sequential( # f encoder
             self.backbone,
             self.projector
        )
        self.predictor = prediction_MLP()
    
    def forward(self, x1, x2, mask):
        x1=data_selfattention(x1,x1,x1,mask)#不加attention为42.69
        x2=data_selfattention(x2,x2,x2,mask)
        z1,z2=self.encoder(x1),self.encoder(x2)
        p1,p2=self.predictor(z1),self.predictor(z2)
        return p1,z2,p2,z1






if __name__ == "__main__":



    model = Plus_Proj_layer()
    x1 = torch.randn((2, 3, 224, 224))
    x2 = torch.randn_like(x1)#创建像x1的大小的张量

    model.forward(x1, x2).backward()
    print("forward backwork check")

    z1 = torch.randn((200, 2560))
    z2 = torch.randn_like(z1)
    import time
    tic = time.time()
    print(cos_similarity(z1, z2, version='original'))
    toc = time.time()
    print(toc - tic)
    tic = time.time()
    print(cos_similarity(z1, z2, version='simplified'))
    toc = time.time()
    print(toc - tic)













