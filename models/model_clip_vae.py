from functools import partial
from transformers import CLIPTextConfig, CLIPVisionConfig
from transformers import CLIPConfig, CLIPModel
from transformers import CLIPVisionModel
from transformers import AutoTokenizer, CLIPTextModel
import torch
from torch import nn
import torch.nn.functional as F

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

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
        config_vision = CLIPVisionConfig()

        self.visual_encoder = CLIPVisionModel.from_pretrained("/workspace/project/clip_base")    
        bert_config = CLIPTextConfig.from_json_file(config['config'])
        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder, config=config_text)
        ############VAE
        zsize = 512
        self.fc1 = nn.Linear(zsize, zsize)   # 4 * 4 is the current size of the image
        self.fc2 = nn.Linear(zsize, zsize)

        # encoder layers
        self.enc_txt_fc = nn.Linear(512, int(0.5 * zsize))
        self.enc_img_fc1 = nn.Linear(768, int(0.5 * zsize))
#         self.enc_img_fc2 = nn.Linear(1024, int(0.5 * zsize))
        
        # decoder layers
        self.dec_txt_fc = nn.Linear(zsize, 512)  #512
        self.dec_img_fc1 = nn.Linear(zsize, 768) #768

        # batch normalizations
        self.enc_txt_bn = nn.BatchNorm1d(num_features=int(0.5 * zsize))
        self.enc_img_bn1 = nn.BatchNorm1d(num_features=int(0.5 * zsize))
#         self.enc_img_bn2 = nn.BatchNorm1d(num_features=int(0.5 * zsize))
        
        self.dec_txt_bn = nn.BatchNorm1d(num_features=512)
        self.dec_img_bn1 = nn.BatchNorm1d(num_features=768)
#         self.dec_img_bn2 = nn.BatchNorm1d(num_features=2048)
        
        # dropout
        self.dropout_txt_enc = nn.Dropout(0.2)
        self.dropout_img_enc = nn.Dropout(0.2)
        self.dropout_txt_dec = nn.Dropout(0.2)
        self.dropout_img_dec = nn.Dropout(0.2)

        ######
        # multi-tasks sub-networks
        self.cls = nn.Linear(zsize, 2)
        # self.cls_head = nn.Sequential(
        #           nn.Linear(512, 2),
        #         )
        # self.cls_head = nn.Sequential(
        #           nn.Linear(1280, 1280),
        #           nn.ReLU(),
        #           nn.Linear(1280, 2)#ve是三分类需要改成二分类
        #         )

        if self.distill:
            pass
    def img_encode(self, x_img):
#         _, x_img = self.resnet_pretrained(x_img)
        x_img = F.relu(self.dropout_img_enc(self.enc_img_bn1(self.enc_img_fc1(x_img))))
#         x_img = F.relu(self.enc_img_fc2(x_img))
        
        return x_img   # [bs, 2048]

    def txt_encode(self, x_txt):

        #x_txt = x_txt.view(x_txt.shape[0], 1024)
        x_txt = F.relu(self.dropout_txt_enc(self.enc_txt_bn(self.enc_txt_fc(x_txt))))
        return x_txt   # [bs, 0.5 * zsize]

    def encode(self, x_img, x_txt):
        
        x_img = self.img_encode(x_img)
        
        x_txt = self.txt_encode(x_txt)
        
        # concate x_img and x_txt
        x = torch.cat((x_txt, x_img), 1)
        
        h1 = self.fc1(x)   # mu
        h2 = self.fc2(x)   # logvar
        return h1, h2
    
    def reparameterize(self, mu, logvar, train=True):
        if train:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
            
    def decode(self, x):
#         x = x.view(x.shape[0], self.zsize)   # flatten

        # Decoding txt
        dec_x_txt = F.relu(self.dropout_txt_dec(self.dec_txt_bn(self.dec_txt_fc(x))))
        
        # Decoding img
        dec_x_img = F.relu(self.dropout_img_dec(self.dec_img_bn1(self.dec_img_fc1(x))))
    #         dec_x_img = F.relu(self.dec_img_fc2(dec_x_img))
    
        return dec_x_img, dec_x_txt 

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


    def forward(self, image, text, targets, alpha=0, train=True):
        
        image_embeds = self.visual_encoder(image) 
        image_embeds = image_embeds.pooler_output
        # image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        
        #image_embeds = image_embeds.last_hidden_state[:,0,:]
        # print(image_embeds.shape)
        if train:
            # print(text)
            output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask, return_dict = True) 
            output = output.pooler_output
            mu, logvar = self.encode(image_embeds, output)
            mu = mu.squeeze()
            logvar = logvar.squeeze()
            z = self.reparameterize(mu, logvar, train)
            prediction = self.cls(z)
            dec_x_img, dec_x_txt = self.decode(z.view(-1, 512))



            # prediction = self.cls_head(torch.cat((image_embeds.pooler_output, output.pooler_output),dim=1))          
            # prediction = self.cls_head(output.pooler_output)                           
            
            loss = F.cross_entropy(prediction, targets)     

            return loss, dec_x_img, dec_x_txt, mu, logvar, image_embeds, output
            
        else:
            output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask, return_dict = True)    
            output = output.pooler_output
            mu, logvar = self.encode(image_embeds, output)
            mu = mu.squeeze()
            logvar = logvar.squeeze()
            z = self.reparameterize(mu, logvar, train)
            prediction = self.cls(z)
            dec_x_img, dec_x_txt = self.decode(z.view(-1, 512))

            #prediction = self.cls_head(torch.cat((image_embeds.pooler_output, output.pooler_output),dim=1))          
            return prediction
 



