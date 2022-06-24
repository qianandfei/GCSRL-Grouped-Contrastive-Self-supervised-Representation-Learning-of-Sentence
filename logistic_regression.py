import torch.nn as nn
import torch


class LogisticRegression(nn.Module):
    def __init__(self, n_features, n_classes):
        super(LogisticRegression, self).__init__()
        #self.match_model=nn.Linear(300,512)
        
        self.model = nn.Linear(n_features, n_classes,bias=True)

    def forward(self, x):
               return self.model(x)
