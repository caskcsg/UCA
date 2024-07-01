from functools import partial
from transformers import CLIPTextConfig, CLIPVisionConfig
from transformers import CLIPConfig, CLIPModel
from transformers import CLIPVisionModel
from transformers import AutoTokenizer, CLIPTextModel
import torch
from torch import nn
import torch.nn.functional as F
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

        self.cls_head = nn.Sequential(
                  nn.Linear(1792, 1792),
                  nn.ReLU(),
                  nn.Linear(1792, 2)#ve是三分类需要改成二分类
                )

            
    def forward(self, image, text, targets, alpha=0, train=True):
        
        image_embeds = self.image_encoder(image).pooler_output #1024
        output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask, return_dict = True).pooler_output #768


        prediction = self.cls_head(torch.cat((image_embeds, output),dim=1))          

        loss = F.cross_entropy(prediction, targets)                
            
        return loss, prediction
 


    # @torch.no_grad()    
    # def copy_params(self):
    #     for model_pair in self.model_pairs:           
    #         for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
    #             param_m.data.copy_(param.data)  # initialize
    #             param_m.requires_grad = False  # not update by gradient    

            
    # @torch.no_grad()        
    # def _momentum_update(self):
    #     for model_pair in self.model_pairs:           
    #         for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
    #             param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                

