from functools import partial
from transformers import CLIPTextConfig, CLIPVisionConfig
from transformers import CLIPConfig, CLIPModel
from transformers import CLIPVisionModel
from transformers import AutoTokenizer, CLIPTextModel
import torch
from torch import nn
import torch.nn.functional as F
from models.MLPProcess import MLPEncoder



class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,     
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        self.distill = config['distill']

        config_text = CLIPTextConfig()

        self.visual_encoder = CLIPVisionModel.from_pretrained("/workspace/project/clip_base")    

        bert_config = CLIPTextConfig.from_json_file(config['config'])

        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder, config=config_text)
        self.mlp_encoder = MLPEncoder(activate='gelu', d_in=[50, 2, 768], d_hiddens=[[50, 2, 768], [10,2,32]], d_outs=[[50, 2, 768], [10,2,32]], dropouts=[0.5,0.5,0.5], bias=False, ln_first=False, res_project=[True,True])
        self.proj = nn.Linear(512, 768)
        self.cls = nn.Linear(64, 2)
        self.cls_head = nn.Sequential(
                  nn.Linear(1280, 1280),
                  nn.ReLU(),
                  nn.Linear(1280, 2)#ve是三分类需要改成二分类
                )

        if self.distill:
            pass
            
    def forward(self, image, text, targets, alpha=0, train=True):
        
        image_embeds = self.visual_encoder(image) 
        # print(image_embeds.last_hidden_state.shape) #768
        if train:
            # print(text)
            output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask, return_dict = True) 
            # print(output.last_hidden_state.shape) #512
            
            x = torch.stack([image_embeds.last_hidden_state , self.proj(output.last_hidden_state)], dim=2)
            x = self.mlp_encoder(x, mask=None)
            fused_features = x.mean(dim=1)
            # print(fused_features.shape)
            # print(fused_features)

            used_features = torch.cat(torch.split(fused_features, 1, dim=1), dim=-1).squeeze(1)
            prediction1 = self.cls(used_features)                

            prediction2 = self.cls_head(torch.cat((image_embeds.pooler_output, output.pooler_output),dim=1))   
            prediction = prediction1 + prediction2
            # prediction = self.cls_head(output.pooler_output)                

            if self.distill: 
                loss = 0
                pass               
            else:
                loss = F.cross_entropy(prediction, targets)                
            return loss 
            
        else:
            output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask, return_dict = True)    
            # prediction = self.cls_head(torch.cat((image_embeds, output),dim=1))    
            # prediction = self.cls_head(output.pooler_output)      
            x = torch.stack([image_embeds.last_hidden_state , self.proj(output.last_hidden_state)], dim=2)
            x = self.mlp_encoder(x, mask=None)
            fused_features = x.mean(dim=1)
            # print(fused_features.shape)
            # print(fused_features)

            used_features = torch.cat(torch.split(fused_features, 1, dim=1), dim=-1).squeeze(1)
            prediction1 = self.cls(used_features)             
            prediction2 = self.cls_head(torch.cat((image_embeds.pooler_output, output.pooler_output),dim=1))      
            prediction = prediction1 + prediction2

            return prediction
