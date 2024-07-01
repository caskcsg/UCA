import torch.nn as nn
import torch
from torch.nn import functional
from module.classifier_latent import SingleClassifier_H, SingleClassifier_T

def convert_to_one_hot(indices, num_classes):
    batch_size = indices.size(0)
    indices = indices.unsqueeze(1)
    one_hot = indices.new_zeros(batch_size, num_classes).scatter_(1, indices, 1).cuda()
    return one_hot

def masked_softmax(logits, mask=None):
    eps = 1e-20
    probs = functional.softmax(logits, dim=1)
    if mask is not None:
        mask = mask.float()
        probs = probs * mask + eps
        probs = probs / probs.sum(1, keepdim=True)
    return probs

def st_gumbel_softmax(logits, temperature=1.0, mask=None):
    eps = 1e-20
    u = logits.data.new(*logits.size()).uniform_()
    gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
    y = logits + gumbel_noise
    y = masked_softmax(logits=y / temperature, mask=mask)
    y_argmax = y.max(1)[1]
    y_hard = convert_to_one_hot(indices=y_argmax, num_classes=y.size(1)).float()
    y = (y_hard - y).detach() + y
    return y

class Latent_Bert_H(nn.Module):
    def __init__(self):
        super(Latent_Bert_H,self).__init__()
        self.softmax=nn.Softmax(dim=1)
        self.temp=0.3
        self.fc_gt=SingleClassifier_H(768,6,0.1)
        self.fc_a=SingleClassifier_H(768,6,0.1)
        
    def forward(self, image, text):
        dis_image = self.softmax(self.fc_a(image))
        dis_text = st_gumbel_softmax(self.softmax(self.fc_gt(text)),self.temp)
        return dis_image, dis_text

class Latent_Bert_T(nn.Module):
    def __init__(self):
        super(Latent_Bert_T,self).__init__()
        self.softmax=nn.Softmax(dim=1)
        self.temp=0.3
        self.fc_gt=SingleClassifier_T(768,6,0.1)
        self.fc_a=SingleClassifier_T(768,6,0.1)
        
    def forward(self, image, text):
        dis_image = self.softmax(self.fc_a(image))
        dis_text = st_gumbel_softmax(self.softmax(self.fc_gt(text)),self.temp)
        return dis_image, dis_text
