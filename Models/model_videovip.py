import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import einops
import clip

from vip import load_vip
from .net.attention import TransformerModel
from .utils.position_embedding import generate_position_embedding


class VideoVIP(nn.Module):
    '''
    eval: model/text_model
    train: output_layer/temporalModelling
    '''
    def __init__(self, args, device, model_name='resnet50', modelid='RN50'):
        super(VideoVIP, self).__init__()
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

        self.temporal = args.temporal
        self.dropout = 0.0 # if args.tfm_layers > 2 else 0.0
        self.tfm_layers = args.tfm_layers
        self.tfm_heads = args.tfm_heads
        self.hidden_size = 1024

        # network module
        self.temporalModelling = TransformerModel(d_model=self.hidden_size, nhead=self.tfm_heads, num_layer=self.tfm_layers, 
                                                        dropout=self.dropout)
       
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
        # temporal modelling
        vFeature = self.get_temporal_frames_feature(vFeature) #(bs,t,1024)

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
