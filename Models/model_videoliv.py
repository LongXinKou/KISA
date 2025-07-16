'''
keyframe annotator v2
video encoder + attention + annotator
'''
import clip
import torch
import einops
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as T
from liv import load_liv
from .net.attention import *
from .utils.position_embedding import generate_position_embedding



class VideoLIV(torch.nn.Module):
    def __init__(self, args, device):
        super(VideoLIV, self).__init__()

        self.device = device
        self.livmodel = load_liv().to(self.device)
        self.livmodel.eval()
        self.transform = T.Compose([T.ToTensor()])

        for paramliv in self.livmodel.parameters():
            paramliv.requires_grad = False

        self.temporal = args.temporal

        self.dropout = 0.0 # if args.tfm_layers > 2 else 0.0
        self.hidden_size = 1024
        self.tfm_layers = args.tfm_layers
        self.tfm_heads = args.tfm_heads

        # network module
        self.temporalModelling = TransformerModel(d_model=self.hidden_size, nhead=self.tfm_heads, num_layer=self.tfm_layers, 
                                                        dropout=self.dropout)


    def forward(self, videos:torch.Tensor, caption):
        '''
        output tFeature and vFeature for training
        input:
            videos:tensor(b,t,c,h,w)
            caption:list(b,t)
        output:
            vFeature:(b,t,c)
            cFeature:(b,t,c)
        '''
        bs,t,_,_,_= videos.shape
        # encode text(no grad)
        cFeature = []
        for i in range(bs):
            cFeature.append(self.get_text_feature(caption[i]))
        cFeature = torch.stack(cFeature, dim=0) #(b,t,1024)
        
        # encode videos
        iFeature = self.get_frames_feature(einops.rearrange(videos, 'b t c h w -> (b t) c h w'))
        vFeature = einops.rearrange(iFeature, '(b t) c -> b t c', t=t) 

        # temporal modelling
        vFeature = self.get_temporal_frames_feature(vFeature) #(bs,t,1024)

        return vFeature, cFeature
    
    def get_text_feature(self, sentences):
        '''
        input:
            sentences:(n,1)
        output:
            text_embedding:(n,c)
        '''
        # pre-process text
        text_tokens = clip.tokenize(sentences).to(self.device)

        # compute LIV  text embedding
        with torch.no_grad():
            text_embedding = self.livmodel(input=text_tokens, modality="text")
        return text_embedding.to(torch.float32)

    def get_frames_feature(self, frames_tensor:torch.Tensor):
        '''
        input:
            frames_tensor:(n,c,h,w)
        output:
            img_embedding:(n,1024)
        '''
        # compute LIV image embedding
        with torch.no_grad():
            img_embedding = self.livmodel(input=frames_tensor, modality="vision")
        
        return img_embedding.to(torch.float32)

    def get_temporal_frames_feature(self, frames_feature, frames_length=None, attention_mask=None, padding_mask=None):
        '''
        input:
            frames_feature:(b,t,c)
        output:
            temporal_frames_feature:(b,t,c)
        example:
            video --> get_frame_feature --> get_temporal_frames_feature
            frames_feature --> get_temporal_frames_feature
        '''
        # add atten_mask
        batch_size, sequence_length, d_model = frames_feature.shape
        pos_embedding = generate_position_embedding(sequence_length, d_model).to(self.device)

        # padding mask
        if frames_length is not None:
            padding_mask = torch.zeros(batch_size, sequence_length).to(self.device)
            for i in range(batch_size):
                padding_mask[i, frames_length[i]:] = 1

        # attention mask
        attention_mask = nn.Transformer.generate_square_subsequent_mask(sequence_length).to(self.device) #tensor(t,t)

        frames_feature = self.temporalModelling(frames_feature, pos_embedding, padding_mask=padding_mask, atten_mask=attention_mask)
        return frames_feature  

    def text_video_similarity(self, text_features, video_features):
        return (text_features @ video_features.T).mean()
    
