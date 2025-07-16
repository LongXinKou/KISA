'''
Predictor(language embedding, historical observation)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from .net.attention import *
from .net.cvae import *


class cvaePredictor(torch.nn.Module):
    def __init__(self, args, device):
        super(cvaePredictor, self).__init__()

        self.device = device
        self.hidden_size = args.hidden_size
        self.attention = args.attention
        self.att_heads = args.att_heads
        self.latent_dim = args.latent_dim

        self.input_size = 1
        self.output_size = args.library_size

        # network module
        if self.attention:
            self.attention = LanguageGuidedAttentionModule(d_model=self.hidden_size, n_head=self.att_heads)
        self.annotator = CVAE(input_dim=self.input_size, condition_dim=self.hidden_size, latent_dim=self.latent_dim, output_dim=self.output_size) #skill index
    
    def forward(self, vFeature, labels, tFeature=None):
        '''
        input:
            vFeature:tensor(b,t,c)
            tFeature:tensor(b,t,c)
            input_skill:tensor(b,t)
        output:
            pred_skill:(b,t,n)
        '''
        bs,t,_ = vFeature.shape
        # language-guided attention
        if self.attention: 
            fusionFeature = self.attention(vFeature, tFeature, tFeature) #(bs,t,1024) 
        else:
            fusionFeature = vFeature #(bs,t,1024) 

        # skill predictor
        fusionFeature = einops.rearrange(fusionFeature, 'b t d -> (b t) d')
        input_skill = labels.view(-1,1) #(bs*t,1)

        reconstructed_skill, mu, logvar = self.annotator(input_skill, fusionFeature)
        
        reconstructed_skill = einops.rearrange(reconstructed_skill, '(b t) c -> b t c', t=t) #(bs,t,n)    

        return reconstructed_skill, mu, logvar

    def predict(self, vFeature, tFeature):
        '''
        input:
            vFeature:tensor(b,t,c)
            tFeature:tensor(b,t,c)
        output:
            reconstructed_skill:(b,t,n)
        '''
        bs,t,_ = vFeature.shape
        # language-guided attention
        fusionFeature = self.attention(vFeature, tFeature, tFeature) #(bs,t,1024)

        # skill predictor
        fusionFeature = einops.rearrange(fusionFeature, 'b t d -> (b t) d')
        reconstructed_skill = self.annotator.generate_data(fusionFeature)
        reconstructed_skill = einops.rearrange(reconstructed_skill, '(b t) c -> b t c', t=t) #(bs,t,n)    
        return reconstructed_skill