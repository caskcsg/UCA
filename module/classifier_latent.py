import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

class SingleClassifier_H(nn.Module):
    def __init__(self,in_dim,out_dim,dropout):
        super(SingleClassifier_H,self).__init__()
        layer=[
            weight_norm(nn.Linear(in_dim,out_dim),dim=None),
            nn.Dropout(dropout,inplace=True)
        ]
        self.main=nn.Sequential(*layer)
        
    def forward(self,x):
        logits=self.main(x)
        return logits   
    
    
class SingleClassifier_T(nn.Module):
    def __init__(self,in_dim,out_dim,dropout):
        super(SingleClassifier_T,self).__init__()
        layer=[
            weight_norm(nn.Linear(in_dim,out_dim),dim=None),
            nn.Dropout(dropout,inplace=True)
        ]
        self.main=nn.Sequential(*layer)
        
    def forward(self,x):
        logits=self.main(x)
        return logits        