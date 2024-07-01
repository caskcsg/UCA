from functools import partial
from transformers import CLIPTextConfig, CLIPVisionConfig
from transformers import CLIPConfig, CLIPModel
from transformers import CLIPVisionModel
from transformers import AutoTokenizer, CLIPTextModel
import torch
from torch import nn
import torch.nn.functional as F
import copy
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus
import math

import numpy as np


def GaussianReParameterize(mean, logvar, n_samples=1):
    r""" Sampling from N(mean,exp(logvar)) through: z = mean + exp(logvar*0.5)*eps
    mean, logvar : [batch_size, dim_z]
    n_samples : int (default 1)
    return: [batch_size, n_samples, dim_z] if n_samples>1 else [batch_size, dim_z]
    """
    mu = mean
    sigma = torch.exp(logvar*0.5)
    if n_samples>1:
        mu_expd = mu.unsqueeze(1).repeat(1, n_samples, 1)
        sigma_expd = sigma.unsqueeze(1).repeat(1, n_samples, 1)
    else:
        mu_expd, sigma_expd = mu, sigma
    eps = torch.zeros_like(sigma_expd).normal_()
    z = eps*sigma_expd + mu_expd
    return z


def LogGaussianPDF(mean, logvar, z):
    r""" Return the log PDF value of z in N(mean,exp(logvar))
    mean, logvar : [*, dim_z]
    z : [*, N, dim_z]
    return: [*, N, dim_z]
    """
    if type(mean) is torch.Tensor:
        mean, logvar = mean.unsqueeze(-2), logvar.unsqueeze(-2)
        return -0.5*np.log(2*np.pi) -0.5*logvar - ((z-mean)**2+1e-6) / (2*torch.exp(logvar)+1e-6)
    elif type(mean) is np.ndarray:
        mean, logvar = np.expand_dims(mean, axis=-2), np.expand_dims(logvar, axis=-2)
        return -0.5*np.log(2*np.pi) -0.5*logvar - ((z-mean)**2+1e-6) / (2*np.exp(logvar)+1e-6)
    return None

def GaussianPDF(mean, logvar, z):
    r""" Return the PDF value of z in N(mean,exp(logvar))
    mean, logvar : [*, dim_z]
    z : [*, N, dim_z]
    return: [*, N, dim_z]
    """
    if type(mean) is torch.Tensor:
        mean, logvar = mean.unsqueeze(-2), logvar.unsqueeze(-2)
        return 1/(np.sqrt(2*np.pi)*torch.exp(logvar*0.5)) * torch.exp(-((z-mean)**2) / (2*torch.exp(logvar)))
    elif type(mean) is np.ndarray:
        mean, logvar = np.expand_dims(mean, axis=-2), np.expand_dims(logvar, axis=-2)
        return 1/(np.sqrt(2*np.pi)*np.exp(logvar*0.5)) * np.exp(-((z-mean)**2) / (2*np.exp(logvar)))
    return None


class Encoder(nn.Module):
    def __init__(self, z_dim=2):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(64, 32), #64
            nn.ReLU(True),
            nn.Linear(32, z_dim * 2),
        )

    def forward(self, x, n_samples=32, agg_size=None):
        # x = x.view(x.size(0), -1)  # Flatten the input
        params = self.net(x)
        mean, logvar = params[:, :self.z_dim], params[:, self.z_dim:]


        batch_size, dim_z = mean.shape
        agg_size = batch_size
        num_chunks = 1
        z = GaussianReParameterize(mean, logvar, n_samples) # z: [batch_size, n_samples, dim_z]
        z1 = z.view(num_chunks, agg_size, n_samples, dim_z) # group into different chunks
        z2 = z1.view(num_chunks, agg_size*n_samples, dim_z) # share inside chunks
        z3 = z2.unsqueeze(1).repeat(1,agg_size,1,1) # copy for every posterior inside chunks
        # [num_chunks, agg_size, agg_size*n_samples, dim_z]
        
        # step2. logq = E_z log(agg-post(z)) = E_z log[1/N q(z_{i,j}|x_{k})], where N is agg_size, i in range(N)
        # q(z_{i,j}|x_{k}) \in R^dim_z is the marginal density on each dimension
        q_ij_given_k = GaussianPDF(
            mean.view(num_chunks, agg_size, dim_z), # [num_chunks, agg_size, dim_z]
            logvar.view(num_chunks, agg_size, dim_z), # [num_chunks, agg_size, dim_z]
            z3, # [num_chunks, agg_size, agg_size*n_samples, dim_z]
        ) # the marginal density on each dimension: [num_chunks, agg_size, agg_size*n_samples, dim_z]
        logq = q_ij_given_k.mean(dim=1).log().mean(dim=1) # the first mean is for agg, and the second is for mc: |b| * M
        # logq: [num_chunks, dim_z]
        
        # step3. logp = E_z log(prior(z)) = E_z log[p(z_{i,j})]
        logp_marginal_i = LogGaussianPDF(
            torch.zeros_like(z2[:,0,:]),
            torch.zeros_like(z2[:,0,:]),
            z2, # [num_chunks, agg_size*n_samples, dim_z]
        )
        logp = logp_marginal_i.mean(dim=1) # this mean is for mc
        # logp: [num_chunks, dim_z]
        
        # step4. return (logq-logp).mean()
        okl = (logq-logp).mean(dim=0) # this mean is for mc: C
        
        return okl.sum(dim=-1)

        


class AmbiguityLearning(nn.Module):
    def __init__(self):
        super(AmbiguityLearning, self).__init__()
        #self.encoding = EncodingPart()
        self.encoder_text = Encoder()
        self.encoder_image = Encoder()

    def forward(self, text_encoding, image_encoding):
        # text_encoding, image_encoding = self.encoding(text, image)
        kl_1_2 = self.encoder_text(text_encoding)
        kl_2_1 = self.encoder_image(image_encoding)
        # z1 = p_z1_given_text.rsample()
        # z2 = p_z2_given_image.rsample()
        # kl_1_2 = p_z1_given_text.log_prob(z1) - p_z2_given_image.log_prob(z1)
        # kl_2_1 = p_z2_given_image.log_prob(z2) - p_z1_given_text.log_prob(z2)
        skl = (kl_1_2 + kl_2_1)/ 2
        skl = nn.functional.sigmoid(skl)
        return skl


class UnimodalDetection(nn.Module):
        def __init__(self, shared_dim=768, prime_dim = 64):
            super(UnimodalDetection, self).__init__()
            self.text_uni = nn.Sequential(
                nn.Linear(768, shared_dim),
                nn.BatchNorm1d(shared_dim),
                nn.ReLU(),
                nn.Linear(shared_dim, prime_dim),
                nn.BatchNorm1d(prime_dim),
                nn.ReLU()
            )
            self.image_uni = nn.Sequential(
                nn.Linear(1024, shared_dim),
                nn.BatchNorm1d(shared_dim),
                nn.ReLU(),
                nn.Linear(shared_dim, prime_dim),
                nn.BatchNorm1d(prime_dim),
                nn.ReLU()
            )

        def forward(self, image_encoding, text_encoding):
            text_prime = self.text_uni(text_encoding)
            image_prime = self.image_uni(image_encoding)
            return image_prime, text_prime



# class CrossModule4Batch(nn.Module):
#     def __init__(self, text_in_dim=64, image_in_dim=64, corre_out_dim=64):
#         super(CrossModule4Batch, self).__init__()
#         self.softmax = nn.Softmax(-1)
#         self.corre_dim = 64
#         self.pooling = nn.AdaptiveMaxPool1d(1)
#         self.c_specific_2 = nn.Sequential(
#             nn.Linear(self.corre_dim, corre_out_dim),
#             nn.BatchNorm1d(corre_out_dim),
#             nn.ReLU()
#         )

#     def forward(self, text, image):
#         text_in = text.unsqueeze(2)
#         image_in = image.unsqueeze(1)
#         corre_dim = text.shape[1]
#         similarity = torch.matmul(text_in, image_in) / math.sqrt(corre_dim)
#         correlation = self.softmax(similarity)
#         correlation_p = self.pooling(correlation).squeeze()
#         correlation_out = self.c_specific_2(correlation_p)
#         return correlation_out


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

        map_dim = 64
        image_map_layers = [nn.Linear(1024, map_dim), nn.Dropout(p=0.2)]
        text_map_layers = [nn.Linear(768, map_dim), nn.Dropout(p=0.2)]
        # for _ in range(1, 1):
        #     image_map_layers.extend([nn.ReLU(), nn.Linear(map_dim, map_dim), nn.Dropout(p=0.2)])
        #     text_map_layers.extend([nn.ReLU(), nn.Linear(map_dim, map_dim), nn.Dropout(p=0.2)])

        self.image_map = nn.Sequential(*image_map_layers)
        self.text_map = nn.Sequential(*text_map_layers)

        pre_output_layers = [nn.Dropout(p=0.4)]
        #pre_output_input_dim = map_dim**2
        #output_input_dim = pre_output_input_dim
        #if 1 >= 1: # first pre-output layer
        pre_output_layers.extend([nn.Linear(4224, 2112), nn.ReLU(), nn.Dropout(p=0.1)]) #544->1088->1152
            #output_input_dim = map_dim
        # for _ in range(1, 1): # next pre-output layers
        #     pre_output_layers.extend([nn.Linear(544, 32), nn.ReLU(), nn.Dropout(p=0.1)])

        self.pre_output = nn.Sequential(*pre_output_layers)
        self.output = nn.Linear(2112, 1)
        self.cross_entropy_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')


        ###############CAFE
        self.ambiguity_module = AmbiguityLearning()
        self.mulproj = UnimodalDetection()

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
        ##########
        ima, tex = self.mulproj(image_embeds, output)
        skl = self.ambiguity_module(ima, tex)
        # skl = self.ambiguity_module(text_features, image_features)
        weight_uni = (1-skl)
        weight_corre = skl
        text_final = weight_uni * text_features
        img_final = weight_uni * image_features

        #############

        image_features = F.normalize(image_features, p=2, dim=1) # [batch_size, d]
        text_features = F.normalize(text_features, p=2, dim=1) # [batch_size, d]
        features = torch.bmm(image_features.unsqueeze(2), text_features.unsqueeze(1)) # [batch_size, d, d]
        features = features.reshape(features.shape[0], -1)  # [batch_size, d*d]  #32*32=1024
        #################
        corre_final = weight_corre * features
        features = torch.cat([text_final, img_final, corre_final], 1) #4096 + 64 + 64 =  4224 
        #####################

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
            
 
