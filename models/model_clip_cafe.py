from functools import partial
from transformers import CLIPTextConfig, CLIPVisionConfig
from transformers import CLIPConfig, CLIPModel
from transformers import CLIPVisionModel
from transformers import AutoTokenizer, CLIPTextModel
import torch
from torch import nn
import torch.nn.functional as F
import math
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus

class SimilarityModule(nn.Module):
    def __init__(self, shared_dim=128, sim_dim=64):
        super(SimilarityModule, self).__init__()
        self.encoding = EncodingPart()
        self.text_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )
        self.image_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )
        self.sim_classifier_dim = sim_dim * 2
        self.sim_classifier = nn.Sequential(
            nn.BatchNorm1d(self.sim_classifier_dim),
            nn.Linear(self.sim_classifier_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, text, image):
        text_encoding, image_encoding = self.encoding(text, image)
        text_aligned = self.text_aligner(text_encoding)
        image_aligned = self.image_aligner(image_encoding)
        sim_feature = torch.cat([text_aligned, image_aligned], 1)
        pred_similarity = self.sim_classifier(sim_feature)
        return text_aligned, image_aligned, pred_similarity


class Encoder(nn.Module):
    def __init__(self, z_dim=2):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, z_dim * 2),
        )

    def forward(self, x):
        # x = x.view(x.size(0), -1)  # Flatten the input
        params = self.net(x)
        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        
        sigma = softplus(sigma) + 1e-7  
        return Independent(Normal(loc=mu, scale=sigma), 1)


class AmbiguityLearning(nn.Module):
    def __init__(self):
        super(AmbiguityLearning, self).__init__()
        #self.encoding = EncodingPart()
        self.encoder_text = Encoder()
        self.encoder_image = Encoder()

    def forward(self, text_encoding, image_encoding):
        # text_encoding, image_encoding = self.encoding(text, image)
        p_z1_given_text = self.encoder_text(text_encoding)
        p_z2_given_image = self.encoder_image(image_encoding)
        z1 = p_z1_given_text.rsample()
        z2 = p_z2_given_image.rsample()
        kl_1_2 = p_z1_given_text.log_prob(z1) - p_z2_given_image.log_prob(z1)
        kl_2_1 = p_z2_given_image.log_prob(z2) - p_z1_given_text.log_prob(z2)
        skl = (kl_1_2 + kl_2_1)/ 2.
        skl = nn.functional.sigmoid(skl)
        return skl


class UnimodalDetection(nn.Module):
        def __init__(self, shared_dim=128, prime_dim = 16):
            super(UnimodalDetection, self).__init__()
            self.text_uni = nn.Sequential(
                nn.Linear(512, shared_dim),
                nn.BatchNorm1d(shared_dim),
                nn.ReLU(),
                nn.Linear(shared_dim, prime_dim),
                nn.BatchNorm1d(prime_dim),
                nn.ReLU()
            )
            self.image_uni = nn.Sequential(
                nn.Linear(768, shared_dim),
                nn.BatchNorm1d(shared_dim),
                nn.ReLU(),
                nn.Linear(shared_dim, prime_dim),
                nn.BatchNorm1d(prime_dim),
                nn.ReLU()
            )

        def forward(self, text_encoding, image_encoding):
            text_prime = self.text_uni(text_encoding)
            image_prime = self.image_uni(image_encoding)
            return text_prime, image_prime


class CrossModule4Batch(nn.Module):
    def __init__(self, text_in_dim=64, image_in_dim=64, corre_out_dim=64):
        super(CrossModule4Batch, self).__init__()
        self.softmax = nn.Softmax(-1)
        self.corre_dim = 64
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.c_specific_2 = nn.Sequential(
            nn.Linear(self.corre_dim, corre_out_dim),
            nn.BatchNorm1d(corre_out_dim),
            nn.ReLU()
        )

    def forward(self, text, image):
        text_in = text.unsqueeze(2)
        image_in = image.unsqueeze(1)
        corre_dim = text.shape[1]
        similarity = torch.matmul(text_in, image_in) / math.sqrt(corre_dim)
        correlation = self.softmax(similarity)
        correlation_p = self.pooling(correlation).squeeze()
        correlation_out = self.c_specific_2(correlation_p)
        return correlation_out



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

        ###############CAFE
        self.ambiguity_module = AmbiguityLearning()
        self.image_proj = nn.Linear(768, 64)
        self.text_proj = nn.Linear(512, 64)
        self.uni_repre = UnimodalDetection()
        self.cross_module = CrossModule4Batch()
        self.classifier_corre = nn.Sequential(
            nn.Linear(64+16+16, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(64, 2)
        )
      
    def forward(self, image, text, targets, alpha=0, train=True):
        
        image_embeds = self.visual_encoder(image)  #768
        image_embeds = image_embeds.pooler_output

        if train:
            output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask, return_dict = True) 
            output = output.pooler_output #512
            am_out = self.text_proj(output)
            am_ima = self.image_proj(image_embeds)

            skl = self.ambiguity_module(am_out, am_ima)
            text_prime, image_prime = self.uni_repre(output, image_embeds)
            correlation = self.cross_module(am_out, am_ima)
            weight_uni = (1-skl).unsqueeze(1)
            weight_corre = skl.unsqueeze(1)
            text_final = weight_uni * text_prime
            img_final = weight_uni * image_prime
            corre_final = weight_corre * correlation
            final_corre = torch.cat([text_final, img_final, corre_final], 1)
            prediction = self.classifier_corre(final_corre)
              
            loss = F.cross_entropy(prediction, targets)                
            return loss 
            
        else:
            output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask, return_dict = True)    
            output = output.pooler_output #512
            am_out = self.text_proj(output)
            am_ima = self.image_proj(image_embeds)

            skl = self.ambiguity_module(am_out, am_ima)
            text_prime, image_prime = self.uni_repre(output, image_embeds)
            correlation = self.cross_module(am_out, am_ima)
            weight_uni = (1-skl).unsqueeze(1)
            weight_corre = skl.unsqueeze(1)
            text_final = weight_uni * text_prime
            img_final = weight_uni * image_prime
            corre_final = weight_corre * correlation
            final_corre = torch.cat([text_final, img_final, corre_final], 1)
            prediction = self.classifier_corre(final_corre)

            return prediction

