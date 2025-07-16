import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_layer, batch_first=True,dropout=None):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layer, num_decoder_layers=num_layer, 
                                          batch_first=batch_first, dropout=dropout)

    def forward(self, x, position_embedding=None, padding_mask=None, atten_mask=None):
        if position_embedding is not None:
            x = x + position_embedding
        output = self.transformer(x, x, tgt_mask=atten_mask, tgt_key_padding_mask=padding_mask, src_key_padding_mask=padding_mask)
        return output

class LanguageGuidedAttentionModule(nn.Module):
    def __init__(self, d_model, n_head=2):
        super(LanguageGuidedAttentionModule, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head)

    def forward(self, q, k, v):
        output, attention_weights = self.attention(q, k, v)
        return output