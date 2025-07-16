import clip
import torch
import einops
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from liv import load_liv


class FLIV(nn.Module):
    def __init__(self, args, device):
        super(FLIV, self).__init__()
        self.args = args
        self.device = device
        self.model = load_liv()
        self.model.to(self.device)
        self.model.eval()

        # freeze model
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.hidden_size = 1024
        self.output_layer = nn.Linear(self.hidden_size, self.hidden_size)
    
    def forward(self, videos, caption):
        bs,t,_,_,_= videos.shape
        # encode text(no grad)
        cFeature = []
        for i in range(bs):
            cFeature.append(self.get_text_feature(caption[i]))
        cFeature = torch.stack(cFeature, dim=0) #(b,t,1024)

        # encode videos
        iFeature = self.get_image_feature(einops.rearrange(videos, 'b t c h w -> (b t) c h w')).to(torch.float32)
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
        # pre-process text
        text_tokens = clip.tokenize(sentences).to(self.device)

        # compute LIV  text embedding
        with torch.no_grad():
            text_embedding = self.model(input=text_tokens, modality="text")
        return text_embedding
    
    def get_image_feature(self, img: torch.Tensor) -> torch.Tensor:
        '''
        input:
            frames_tensor:(n,c,h,w)
        output:
            img_embedding:(n,1024)
        '''
        # encode videos
        image_inputs = img
        # compute LIV image embedding
        img_embedding = self.model(input=image_inputs, modality="vision")
        
        return img_embedding
    
    def txt_to_img_similarity(self, tFeature, vFeature):
        '''
        input:
            tFeature:(n,d)
            vFeature:(t,d)
        output:
            similarity:(t,n)
        '''
        similarity = torch.matmul(vFeature, tFeature.t())
        return similarity
    
    def get_reward(self, vFeature, eFeature=None):
        '''
        input:
            vFeature:(t,d)
            eFeature:(,d)
        output:
            reward:(t)
        '''
        def sim(tensor1, tensor2):
            tensor1 = tensor1 / tensor1.norm(dim=-1, keepdim=True)
            tensor2 = tensor2 / tensor2.norm(dim=-1, keepdim=True)
            d = F.cosine_similarity(tensor1, tensor2)
            return d
        # compare with goal image
        if eFeature is None:
            eFeature = vFeature[-1]

        eFeature = torch.unsqueeze(eFeature, dim=0)
        eFeature_t = eFeature.repeat(vFeature.shape[0], 1) #(t,d)
        with torch.no_grad():
            reward = sim(eFeature_t, vFeature)
        return reward
