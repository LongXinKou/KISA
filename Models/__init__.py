import torch

from .model_videoliv import VideoLIV
from .model_liv import LIV
from .model_fliv import FLIV
from .model_videor3m import VideoR3M
from .model_r3m import R3M
from .model_fr3m import FR3M
from .model_videoclip import VideoCLIP
from .model_clip import CLIP
from .model_videovip import VideoVIP
from .model_vip import VIP
from .model_fvip import FVIP
from .predictor_cvae import cvaePredictor
from .predictor_mlp import mlpPredictor

def count_parameters(model, pretrain=False):
    if not pretrain:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def load_vlm(args, visual_representation, model_path=None, pretrain=False):
    pre_train_list = ["vliv", "vclip", "vr3m", "vvip", "fliv", "fvip", "fr3m"]

    if pretrain and visual_representation in pre_train_list:
        checkpoint = torch.load(model_path, map_location=args.device)
        if 'params' in checkpoint:
            checkpoint = checkpoint['params']
            for k,v in checkpoint.items():
                if k == 'hidden_size':
                    args.hidden_size = v
                elif k == 'tfm_heads':
                    args.tfm_heads = v
                elif k == 'tfm_layers':
                    args.tfm_layers = v

    # Initialize model
    if visual_representation == "vliv":
        model = VideoLIV(args, device=args.device)
    elif visual_representation == 'vclip':
        model = VideoCLIP(args, device=args.device)
    elif visual_representation == 'vr3m':
        model = VideoR3M(args, device=args.device)
    elif visual_representation == 'vvip':
        model = VideoVIP(args, device=args.device)
    elif visual_representation == "liv":
        model = LIV(args, device=args.device)
    elif visual_representation == "clip":
        model = CLIP(args, device=args.device)
    elif visual_representation == "r3m":
        model = R3M(args, device=args.device)
    elif visual_representation == "vip":
        model = VIP(args, device=args.device)
    elif visual_representation == "fliv":
        model = FLIV(args, device=args.device)
    elif visual_representation == "fr3m":
        model = FR3M(args, device=args.device)
    elif visual_representation == "fvip":
        model = FVIP(args, device=args.device)
    
    # Load model
    if pretrain and visual_representation in pre_train_list:
        checkpoint = torch.load(model_path, map_location=args.device)
        if 'model_state_dict' in checkpoint:
            checkpoint = checkpoint['model_state_dict']
        model.load_state_dict(checkpoint)
        
        # for param in model.parameters():
        #     param.requires_grad = False

    parameters_count_grad = count_parameters(model)
    print(f'parameters_count_grad: {parameters_count_grad}')
    parameters_count_all = count_parameters(model, pretrain=True)
    print(f'parameters_count_all: {parameters_count_all}')
    
    return model

def load_predictor(args, hidden_size, output_size):
    if args.predictor == 'mlp':
        predictor_model = mlpPredictor(device=args.device, hidden_size=hidden_size, output_size=output_size)
    elif args.predictor == 'cvae':
        predictor_model = cvaePredictor(args, device=args.device)
    
    return predictor_model