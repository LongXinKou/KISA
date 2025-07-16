# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import *
from utils import *

from train import train_epoch,test_epoch,train_pred_epoch,test_pred_epoch
from Dataset.keyframe_dataset import load_data
from Models import load_vlm, load_predictor
from config import main_config

# Training video encoder
def train_vlm(args):
    torch.autograd.set_detect_anomaly(True)

    # config
    work_dir = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
    save_dir = os.path.join('result/vlm', work_dir)
    args.save_dir = save_dir
    os.makedirs(args.save_dir, exist_ok=True)
    
    logger = SummaryWriter(logdir=args.save_dir)

    # initialize dataloader
    train_loader = load_data(args.batch_size, args.num_workers, data_dir=args.train_path, shuffle=True, version=args.version)
    test_loader = load_data(args.batch_size, args.num_workers, data_dir=args.test_path, shuffle=True, version=args.version)
    print('dataloader initialized!')
    
    # initialize model
    model = load_vlm(args, visual_representation=args.visual_representation, model_path=args.model_path, pretrain=args.pretrain).to(args.device)
    print('model initialized!')

    # load pre-train model
    if args.pretrain:
        last_epoch = int(args.model_path.split('.')[0].split('_')[-1])
    else:
        last_epoch = -1

    # initialize training
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay) 
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,  
                                                                        T_0=int(args.decay_steps), 
                                                                        eta_min=args.lr*0.01, 
                                                                        last_epoch=last_epoch)
    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        criterion.cuda()

    # skill library
    library_path = os.path.dirname(args.train_path)
    with open(os.path.join(library_path,'skill.json'), 'r') as json_file:
        task_information = json.load(json_file)
    skill_library = task_information["skill"]
    slFeature = model.get_text_feature(skill_library) #(n,c)

    for iteration in range(last_epoch+1, args.num_iteration):
        train_epoch(args, train_loader, model, optimizer, lr_scheduler, iteration, logger, slFeature)

        if (iteration+1) % args.save_model_every_n_steps == 0:
            test_epoch(args, test_loader, model, iteration+1, logger, slFeature)
            save_checkpoint(
                model_state = model.state_dict(), optim_state=optimizer.state_dict(), is_best=False, step = iteration+1, args=args,name='vencoder_'
            )
        
        print(f"epoch{iteration+1} finished")

# Training Skill Predictor
def train_predictor(args):
    torch.autograd.set_detect_anomaly(True)

    # config
    work_dir = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
    save_dir = os.path.join('result/predictor', work_dir)
    args.save_dir = save_dir
    os.makedirs(args.save_dir, exist_ok=True)
    
    logger = SummaryWriter(logdir=args.save_dir)

    # initialize dataloader
    train_loader = load_data(args.batch_size, args.num_workers, data_dir=args.train_path, shuffle=True, version=args.version)
    test_loader = load_data(args.batch_size, args.num_workers, data_dir=args.test_path, shuffle=True, version=args.version)
    print('dataloader initialized!')
    
    # model1: vlm(freeze)
    model = load_vlm(args, pretrain=True).to(args.device)
    model.eval()
    print('model initialized!')

    # skill library
    library_path = os.path.dirname(args.train_path)
    with open(os.path.join(library_path,'skill.json'), 'r') as json_file:
        task_information = json.load(json_file)
    skill_library = task_information["skill"]
    slFeature = model.get_text_feature(skill_library) #(n,c)
    args.library_size = len(skill_library)

    # model 2: skill predictor
    predictor_model = load_predictor(args, hidden_size=model.hidden_size, output_size=args.library_size).to(args.device)
    predictor_model.train()

    # initialize training
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, predictor_model.parameters()), lr=args.lr, weight_decay=args.weight_decay) 
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,  
                                                                        T_0=int(args.decay_steps), 
                                                                        eta_min=args.lr*0.01, 
                                                                        last_epoch=-1)
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion.cuda()

    args.start_iter = 0

    for iteration in range(args.start_iter, args.num_iteration):
        train_pred_epoch(args, train_loader, model, predictor_model, optimizer, lr_scheduler, iteration, logger, skill_library=slFeature, criterion=criterion)

        if (iteration+1) % args.save_model_every_n_steps == 0:
            test_pred_epoch(args, test_loader, model, predictor_model, iteration+1, logger, skill_library=slFeature, criterion=criterion)
            save_checkpoint(
                state = predictor_model.state_dict(), is_best=False, step = iteration+1, args=args,name='predictor_'
            )
        
        print(f"epoch{iteration+1} finished")


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if args.train_mode == 'vlm':
        print(f'{args.visual_representation} VLM Training')
        train_vlm(args)
        print(args)
    else:
        print(f'{args.visual_representation} Predictor Training')
        train_predictor(args)
        print(args)

if __name__ == '__main__':    
    args = main_config()
    make_dir(args)
    main(args)
    