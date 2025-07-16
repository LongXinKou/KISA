'''
Predictor(language embedding, historical observation)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from .net.attention import *
from .net.mlp import *


class mlpPredictor(torch.nn.Module):
    def __init__(self, device, hidden_size, output_size, attention=False, att_heads=2):
        super(mlpPredictor, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.attention = attention
        self.att_heads = att_heads
        
        # network module
        if self.attention:
            self.attention = LanguageGuidedAttentionModule(d_model=self.hidden_size, n_head=self.att_heads)
        self.annotator = mlpAnnotator(input_dim=self.hidden_size, output_dim=self.output_size)
    
    def forward(self, vFeature, tFeature=None):
        '''
        input:
            vFeature:tensor(b,t,c)
            tFeature:tensor(b,t,c)
        output:
            pred_skill:(b,t,c)
        '''
        bs,t,_ = vFeature.shape
        # language-guided attention
        if self.attention: 
            fusionFeature = self.attention(vFeature, tFeature, tFeature) #(bs,t,1024) 
        else:
            fusionFeature = vFeature

        # skill predictor
        pred_skill = self.annotator(fusionFeature) #(bs,t,1024) 
        return pred_skill
