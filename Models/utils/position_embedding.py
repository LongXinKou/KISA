import torch

def generate_position_embedding(seq_len, d_model):
    position = torch.arange(0, seq_len).unsqueeze(1)
    div_term = torch.pow(10000, torch.arange(0, d_model, 2).float() / d_model)
    div_term = div_term.view(1, -1)
    pos_embedding = torch.cat((torch.sin(position / div_term), torch.cos(position / div_term)), dim=1)
    return pos_embedding.unsqueeze(0)