from functools import partial
from transformers import CLIPTextConfig, CLIPVisionConfig
from transformers import CLIPConfig, CLIPModel
from transformers import CLIPVisionModel
from transformers import AutoTokenizer, CLIPTextModel
import torch
from torch import nn
import torch.nn.functional as F
from models.mmdynamics import MMDynamic



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

        dim_list_image = [768] * config['batch_size_train']
        self.mmdynamic =  MMDynamic(in_dim=dim_list_image, hidden_dim=[768], num_class=2, dropout=0.5)


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
            
            image_MMLoss, _ = self.mmdynamic(image_embeds.last_hidden_state.transpose(0,1), targets)
            
            prediction = self.cls_head(torch.cat((image_embeds.pooler_output, output.pooler_output),dim=1))   
            # prediction = self.cls_head(output.pooler_output)                
            # print(image_MMLoss)
            if self.distill: 
                loss = 0
                pass               
            else:
                loss = F.cross_entropy(prediction, targets)                
            return loss + 0.01 * image_MMLoss
            
        else:
            output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask, return_dict = True)    
            # prediction = self.cls_head(torch.cat((image_embeds, output),dim=1))    
            # prediction = self.cls_head(output.pooler_output)      
            prediction = self.cls_head(torch.cat((image_embeds.pooler_output, output.pooler_output),dim=1))      

            return prediction
