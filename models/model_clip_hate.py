from functools import partial
from transformers import CLIPTextConfig, CLIPVisionConfig
from transformers import CLIPConfig, CLIPModel
from transformers import CLIPVisionModel
from transformers import AutoTokenizer, CLIPTextModel
import torch
from torch import nn
import torch.nn.functional as F
import torchmetrics
import copy


class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,     
                 ):
        super().__init__()
        
        self.clip = CLIPModel.from_pretrained("/workspace/project/clip_large")
        
        self.image_encoder = copy.deepcopy(self.clip.vision_model)
        self.text_encoder = copy.deepcopy(self.clip.text_model)

        map_dim = 32
        image_map_layers = [nn.Linear(1024, map_dim), nn.Dropout(p=0.2)]
        text_map_layers = [nn.Linear(768, map_dim), nn.Dropout(p=0.2)]
        for _ in range(1, 1):
            image_map_layers.extend([nn.ReLU(), nn.Linear(map_dim, map_dim), nn.Dropout(p=0.2)])
            text_map_layers.extend([nn.ReLU(), nn.Linear(map_dim, map_dim), nn.Dropout(p=0.2)])

        self.image_map = nn.Sequential(*image_map_layers)
        self.text_map = nn.Sequential(*text_map_layers)

        pre_output_layers = [nn.Dropout(p=0.4)]
        pre_output_input_dim = map_dim**2
        output_input_dim = pre_output_input_dim
        if 1 >= 1: # first pre-output layer
            pre_output_layers.extend([nn.Linear(pre_output_input_dim, map_dim), nn.ReLU(), nn.Dropout(p=0.1)])
            output_input_dim = map_dim
        for _ in range(1, 1): # next pre-output layers
            pre_output_layers.extend([nn.Linear(map_dim, map_dim), nn.ReLU(), nn.Dropout(p=0.1)])

        self.pre_output = nn.Sequential(*pre_output_layers)
        self.output = nn.Linear(output_input_dim, 1)
        self.cross_entropy_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

        # self.acc = torchmetrics.Accuracy()
        # self.auroc = torchmetrics.AUROC()
        # self.precision_score = torchmetrics.Precision()
        # self.recall = torchmetrics.Recall()
        # self.f1 = torchmetrics.F1Score()
        
        for _, p in self.image_encoder.named_parameters():
            p.requires_grad_(False)

        for _, p in self.text_encoder.named_parameters():
            p.requires_grad_(False)

        del self.clip
            
    def forward(self, image, text, targets, alpha=0, train=True):
        
        image_embeds = self.image_encoder(image).pooler_output 
        output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask, return_dict = True).pooler_output

        # print(image_embeds.shape) #1024
        # print(output.shape) #768
        image_features = self.image_map(image_embeds)
               
        text_features = self.text_map(output)
        image_features = F.normalize(image_features, p=2, dim=1) # [batch_size, d]
        text_features = F.normalize(text_features, p=2, dim=1) # [batch_size, d]
        features = torch.bmm(image_features.unsqueeze(2), text_features.unsqueeze(1)) # [batch_size, d, d]
        features = features.reshape(features.shape[0], -1)  # [batch_size, d*d]
        features = self.pre_output(features)
        logits = self.output(features).squeeze(dim=1) # [batch_size, 1(or)n]

        preds_proxy = torch.sigmoid(logits)
        preds = (preds_proxy >= 0.5).long()
        
        output = {}

        output['loss'] = self.cross_entropy_loss(logits, targets.float())
        #output['accuracy'] = self.acc(preds, targets)
        #output['auroc'] = self.auroc(preds_proxy, targets)
        output['preds'] = preds
        output['preds_proxy'] = preds_proxy

        return output 
            
 
