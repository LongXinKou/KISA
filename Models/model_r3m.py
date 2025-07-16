import torch
import torch.nn as nn
import torchvision.transforms as T
import einops
import numpy as np
import clip
from r3m import load_r3m


r3m_hidden_dim = {'resnet18':512, 'resnet34':512, 'resnet50':2048}

class R3M(nn.Module):
    """docstring for FeatureExtractor."""

    def __init__(self, args, device, model_name='resnet50', modelid='RN50'):
        super(R3M, self).__init__()
        self.args = args
        self.device = device
        # vision model:r3m
        self.model = load_r3m(model_name)
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        # text model:clip
        self.text_model, _ = clip.load(modelid, device=device)
        self.text_model.to(self.device)
        self.text_model.eval()
        for param in self.text_model.parameters():
            param.requires_grad = False

        self.hidden_size = 1024
        
        # project(no training)
        self.r3m_hidden_dim = r3m_hidden_dim[model_name]
        self.output_layer = nn.Linear(self.r3m_hidden_dim, self.hidden_size) #2048->1024
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

       
    def forward(self, videos, caption):
        bs,t,_,_,_= videos.shape
        # encode text(no grad)
        cFeature = []
        for i in range(bs):
            cFeature.append(self.get_text_feature(caption[i]))
        cFeature = torch.stack(cFeature, dim=0) #(b,t,1024)

        # encode videos
        iFeature = self.get_frames_feature(einops.rearrange(videos, 'b t c h w -> (b t) c h w')).to(torch.float32)
        iFeature = self.output_layer(iFeature)
        vFeature = einops.rearrange(iFeature, '(b t) c -> b t c', t=t) #(b,t,c)            

        return vFeature, cFeature
    
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
    
    def get_frames_feature(self, frames_tensor:torch.Tensor):
        with torch.no_grad():
            image_feature = self.model(frames_tensor*255.0) # R3M expects image input to be [0-255]

        return image_feature 
    

