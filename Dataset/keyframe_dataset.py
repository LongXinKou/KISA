import os
import json
import gymnasium as gym
import numpy as np
import h5py
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# =====================v0=========================
# image trajectory
class KeyframeDataset(Dataset):
    def __init__(self, dataset_file: str, load_count=-1) -> None:
        self.dataset_file = dataset_file 
        with h5py.File(self.dataset_file, "r") as hf:
            group_1 = hf["video_frame"]
            video_frames = [group_1[f"array_{i}"][:] for i in range(len(group_1))]

            group_2 = hf["label"] 
            labels = [group_2[f"array_{i}"][:] for i in range(len(group_2))]

            group_3 = hf["caption"] 
            captions = [group_3[f"array_{i}"][:].astype('str') for i in range(len(group_3))]

            
        self.observations = video_frames
        self.labels = labels
        self.captions = captions
        self.total_frames = len(self.observations)
        if load_count == -1:
            self.load_count = len(self.observations)
        else:
            self.load_count = load_count
        
    def __len__(self):
        return self.load_count

    def __getitem__(self, index):
        video = self.observations[index]
        video = np.transpose(video,[0,3,1,2]) # array(t,3,h,w)
        video = video / 255.
        label = self.labels[index] #array(t,1)
        caption = self.captions[index] #list(k)
        return video, label, caption

# 等长视频预处理
def collate_fn(data):
    video_list = []
    label_list = []
    caption_list = []
    for video, label, caption in data:
        video_list.append(torch.tensor(video).float())
        label_list.append(torch.tensor(label).long())
        caption_list.append(caption)
        
    video_tensor = torch.stack(video_list, dim=0) #tensor(bs,t,3,h,w)
    label_tensor = torch.stack(label_list, dim=0) #tensor(bs,t,1)
    return video_tensor, label_tensor, caption_list

# =====================v1=========================
class KeyframeDataset_v1(Dataset):
    def __init__(self, dataset_file: str, load_count=-1) -> None:
        self.dataset_file = dataset_file 
        with h5py.File(self.dataset_file, "r") as hf:
            group_1 = hf["video_frame"]
            video_frames = [group_1[f"array_{i}"][:] for i in range(len(group_1))]

            group_2 = hf["label"] 
            labels = [group_2[f"array_{i}"][:] for i in range(len(group_2))]

            group_3 = hf["caption"] 
            captions = [group_3[f"array_{i}"][:].astype('str') for i in range(len(group_3))]

            group_4 = hf["gtscore"] 
            gtscores = [group_4[f"array_{i}"][:] for i in range(len(group_4))]

            
        self.observations = video_frames
        self.labels = labels
        self.captions = captions
        self.gtscores = gtscores
        self.total_frames = len(self.observations)
        if load_count == -1:
            self.load_count = len(self.observations)
        else:
            self.load_count = load_count
        
    def __len__(self):
        return self.load_count

    def __getitem__(self, index):
        video = self.observations[index]
        video = np.transpose(video,[0,3,1,2]) # array(t,3,h,w)
        video = video / 255.
        label = self.labels[index] #array(t,1)
        caption = self.captions[index] #list(k)
        gtscore = self.gtscores[index] #array(t,1)
        return video, label, caption, gtscore

# 等长视频预处理
def collate_fn_v1(data):
    video_list = []
    label_list = []
    caption_list = []
    gtscore_list = []
    for video, label, caption, gtscore in data:
        video_list.append(torch.tensor(video).float())
        label_list.append(torch.tensor(label).long())
        caption_list.append(caption)
        gtscore_list.append(torch.tensor(gtscore).float())
        
    video_tensor = torch.stack(video_list, dim=0) #tensor(bs,t,3,h,w)
    label_tensor = torch.stack(label_list, dim=0) #tensor(bs,t,1)
    gtscore_tensor = torch.stack(gtscore_list, dim=0) #tensor(bs,t,1)
    return video_tensor, label_tensor, caption_list, gtscore_tensor

def load_data(batchsize:int, numworkers:int, data_dir:str, shuffle=False, version=0):# -> tuple[DataLoader]:
    if version == 0:
        data = KeyframeDataset(data_dir)
        dataloader = DataLoader(
                        data,
                        batch_size = batchsize,
                        num_workers = numworkers,
                        drop_last = True,
                        collate_fn= collate_fn,
                        shuffle = shuffle
                    )
    else:
        data = KeyframeDataset_v1(data_dir)
        dataloader = DataLoader(
                        data,
                        batch_size = batchsize,
                        num_workers = numworkers,
                        drop_last = True,
                        collate_fn= collate_fn_v1,
                        shuffle = shuffle
                    )
    
    return dataloader


if __name__ == '__main__':
    train_loader = load_data(2, 2, data_dir="/home/rl/franka_kitchen_demos/train_dataset.h5", version=1)
    with tqdm(train_loader, dynamic_ncols=True) as tqdmDataLoader:
        for i, (frames,labeles,captions,gtscores) in enumerate(tqdmDataLoader):
            frames = frames.float()