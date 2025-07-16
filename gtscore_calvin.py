import pickle
import numpy as np
import torch
import h5py
import os
import json

from tqdm import tqdm
from Models import load_vlm
from config import gtscore_config
from utils import Calculate_Similarity, display_similarity
from Models.model_liv import LIV

def load_data(dataset_file):
    with h5py.File(dataset_file, "r") as hf:
        group_1 = hf["video_frame"]
        frames = [group_1[f"array_{i}"][:] for i in range(len(group_1))]

        group_2 = hf["key_frame"]
        keyframe_indices = [group_2[f"array_{i}"][:] for i in range(len(group_2))]

        group_3 = hf["label"]
        labels = [group_3[f"array_{i}"][:] for i in range(len(group_3))]

    return frames, labels, keyframe_indices

def construct_dataset(file_path, embedding_policy):
    frames, labels, keyframe_indices = load_data(dataset_file=file_path)

    gtscore_list = []
    for i in tqdm(range(len(frames)), desc="generate gtscore"):
        video = frames[i]
        keyframe_idx = keyframe_indices[i]
        label = np.zeros(video.shape[0], dtype=np.uint8)
        for i,keyframe in enumerate(keyframe_idx):
            if i == 0:
                label[:keyframe+1] = i
            else:
                label[keyframe_idx[i-1]+1:keyframe+1] = i
        label[keyframe:] = i
        label = torch.tensor(label).to(torch.int64).to(args.device)

        # compute gtscore
        video = np.transpose(video,[0,3,1,2]) 
        video = video / 255.
        video = torch.tensor(video).to(args.device)
        vFeature = embedding_policy.get_image_feature(video)  #tensor(t, d)

        sequence_reward = []
        for keyframe in keyframe_idx:
            kFeature = vFeature[keyframe]
            sequence_reward.append(embedding_policy.get_reward(vFeature, kFeature)) 
        sequence_reward_tensor = torch.stack(sequence_reward, dim=0)
        sequence_reward_tensor = torch.gather(sequence_reward_tensor, 0, label.view(1, -1))
        sequence_reward_array = sequence_reward_tensor.detach().cpu().numpy().reshape(-1,1) # array(t,1)
        
        keyframe_idx[-1] = sequence_reward_array.shape[0]-1
        segments = np.split(sequence_reward_array, keyframe_idx)
        sequence_reward_array = np.concatenate([((seg - seg.min()) / (seg.max() - seg.min() + 1e-6)) for seg in segments])
        gtscore_list.append(sequence_reward_array)

    # save gtscore
    with h5py.File(file_path, 'a') as hf:
        group_4 = hf.create_group("gtscore")
        for i, arr in enumerate(gtscore_list):
            group_4.create_dataset(f"array_{i}", data=arr)

def Generate_gtscore_png(args, file_path, model, skill_library, seed_list):
    with h5py.File(file_path, "r") as hf:
        group_1 = hf["video_frame"]
        frames = [group_1[f"array_{i}"][:] for i in range(len(group_1))]

        group_2 = hf["key_frame"]
        keyframe_indices = [group_2[f"array_{i}"][:] for i in range(len(group_2))]

        group_3 = hf["gtscore"]
        gtscores = [group_3[f"array_{i}"][:] for i in range(len(group_3))]

        group_4 = hf["caption"]
        captions = [group_4[f"array_{i}"][:].astype('str') for i in range(len(group_4))]
    
    for seed in seed_list:
        video = frames[seed]
        keyframe_idx = keyframe_indices[seed]
        caption = [captions[seed]]
        unique_caption = np.unique(caption)
        gtscore = gtscores[seed]
        slFeature = model.get_text_feature(skill_library) #(n,c)

        # preprocess
        video = np.transpose(video,[0,3,1,2])
        video = torch.tensor(video).to(args.device)
        video = video / 255.

        # compute score
        video = torch.unsqueeze(video, dim=0)
        vFeature, cFeature = model(video, caption=caption) #tensor(b,t,c)/(b,t,c)
        logits = []
        for Feature in slFeature:
            tFeature = torch.unsqueeze(torch.unsqueeze(Feature, 0), 0).repeat(1, vFeature.shape[1], 1) #tensor(1,t,c)
            logits.append(Calculate_Similarity(vFeature, tFeature)) #tensor(1,t,1)
        logits = torch.stack(logits, dim=2) #tensor(1,t,k,1)
        logits = torch.squeeze(logits, dim=-1)
        pred_reward, pred_idx = logits.max(dim=2) #tensor(1,t)/tensor(1,t)

        pred_reward_array = pred_reward.detach().cpu().numpy().reshape(-1,1) # array(t,1)
        
        # save confidence score png
        png_dir2 = f'image/gtscore/{args.visual_representation}_{seed}.png'
        display_similarity(y_pred=pred_reward_array, gtscore=gtscore, png_save_dir=png_dir2, subgoal_index=keyframe_idx)


if __name__ == '__main__':
    args = gtscore_config()

    library_path = os.path.dirname(args.file_path)
    with open(os.path.join(library_path,'skill.json'), 'r') as json_file:
        task_information = json.load(json_file)
    skill_library = task_information["skill"]

    # gtscore png
    # model = load_vlm(args, visual_representation=args.visual_representation, model_path=args.model_path, pretrain=True).to(args.device)
    # seed_list = np.arange(0, 150)
    # Generate_gtscore_png(args, args.file_path, model, skill_library, seed_list=seed_list)

    # construct dataset
    model = LIV(args, device=args.device).to(args.device)
    files = ['train', 'test']
    for file in files:
        file_path = f"/data/ubuntu/VideoRLCS/dataset/calvin/{file}_dataset2.h5"
        print(file)
        construct_dataset(file_path, model)

    