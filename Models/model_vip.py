import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import einops
import clip

from vip import load_vip
from .net.attention import TransformerModel
from .utils.position_embedding import generate_position_embedding

vip_hidden_dim = {'resnet18':512, 'resnet34':512, 'resnet50':2048}

class VIP(nn.Module):
    '''
    eval: model/text_model
    train: output_layer/temporalModelling
    '''
    def __init__(self, args, device, model_name='resnet50', modelid='RN50'):
        super(VIP, self).__init__()
        self.args = args
        self.device = device
        # vision model:vip
        self.model = load_vip(model_name).to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        # text model:clip
        self.text_model, _ = clip.load(modelid, device=device)
        self.text_model.to(self.device)
        self.text_model.eval()
        for param in self.text_model.parameters():
            param.requires_grad = False

        self.vip_hidden_dim = vip_hidden_dim[model_name]
        self.hidden_size = 1024
       
    def forward(self, videos:torch.Tensor, caption):
        bs,t,_,_,_= videos.shape
        # encode text(no grad)
        cFeature = []
        for i in range(bs):
            cFeature.append(self.get_text_feature(caption[i]))
        cFeature = torch.stack(cFeature, dim=0) #(b,t,1024)
        
        # encode videos
        iFeature = self.get_frames_feature(einops.rearrange(videos, 'b t c h w -> (b t) c h w'))
        vFeature = einops.rearrange(iFeature, '(b t) c -> b t c', t=t) 

        return vFeature, cFeature
    
    def get_frames_feature(self, frames_tensor:torch.Tensor):
        with torch.no_grad():
            image_feature = self.model(frames_tensor)

        return image_feature 
    
    def get_text_feature(self, sentences):
        '''
        input:
            sentences:(n,)
        output:
            text_embedding:(n,c)
        '''
        text_tokens = clip.tokenize(sentences).to(self.device) #tensor(n,77)
        with torch.no_grad():
            text_embedding = self.text_model.encode_text(text_tokens)

        return text_embedding 
