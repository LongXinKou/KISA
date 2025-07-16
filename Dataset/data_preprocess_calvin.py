import pickle
import os
import json
import numpy as np
import h5py

from collections import Counter
from tqdm import tqdm

def data_preprocess(file_path):
    '''
    output:
        frames: array(100,224,224,3)
        new_keyframe: array(k,)
        label: array(100,)
    '''
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    frames = data['video'] 
    label = np.array(data['skill_index'], dtype=np.uint8)
    task_list = data['skill_name'][:data['finished_task_number']]

    json_file_path = os.path.join(os.path.dirname(os.path.dirname(file_path)), 'skill.json')
    with open(json_file_path, 'r') as json_file:
        task_information = json.load(json_file)
    skill_library = task_information["skill"]
    
    # label generation 
    task_id_list, caption_list = [], []
    for task in task_list:
        caption = task_information[task][-1]
        task_id_list.append(skill_library.index(caption))
        caption_list.append(caption)
    task_id = np.array(task_id_list, dtype=np.uint8)
    label = task_id[label.flatten()].reshape(label.shape) # array(t,)
    
    # keyframe generation
    keyframe_idx = []
    for task_id in task_id_list:
        keyframe_idx.append(np.where(label == task_id)[0][-1])
    keyframe_idx = np.array(keyframe_idx)
    
    # video sample
    sampled_video, new_keyframe, sampled_label, sampled_caption = video_sampling(frames, label, key_frame_idx=keyframe_idx, task_caption=caption_list)

    return sampled_video, new_keyframe, sampled_label, sampled_caption

# def video_sampling(video_frame, label, target_length=150, interval=2, key_frame_idx=None, task_caption=None):
def video_sampling(video_frame, label, target_length=100, interval=2, key_frame_idx=None, task_caption=None): # rebuttle
    '''
    input:
        video_frame:array(t,224,224,3)
        label:array(t,)
    output:
        sampled_frame
        new_keyframe
        sampled_label
    '''
    # target length video sample
    num_keyframe = key_frame_idx.shape[0]
    interval = video_frame.shape[0] / (target_length - num_keyframe)
    sample_indices = [int(i * interval) for i in range(target_length - num_keyframe)]
    for i in key_frame_idx:
        if i not in sample_indices:
            sample_indices.append(i)
        else:
            sample_indices.append(i-1)
    sample_indices = np.sort(sample_indices)
    sampled_video = video_frame[sample_indices] #array(target_length,128,128,3)
    sampled_label = label[sample_indices] #array(target_length,)
    new_keyframe = np.array([np.where(sample_indices == i)[0][0] for i in key_frame_idx], dtype=np.int32) #array(k)
    caption = caption_generation(new_keyframe, target_length, task_caption)

    return sampled_video, new_keyframe, sampled_label, caption

def caption_generation(keyframe_index, t, task_caption):
    '''
    input:
        keyframe_index_list:array(k)
        task_caption:list(k)
    output:
        caption:list(n,k)
    '''
    caption = []
    num_keyframe = keyframe_index.shape[0]
    pre = 0
    for idx in range(num_keyframe):
        caption.extend([task_caption[idx] for _ in range(keyframe_index[idx]-pre+1)])
        pre = keyframe_index[idx]+1
    caption.extend([task_caption[idx] for _ in range(t-pre)])

    return caption

# TODO train/test dataset
def construct_train_dataset(current_folder_path, train_ratio=0.8):
    pkl_files = [file for file in os.listdir(current_folder_path) if file.endswith('.pkl')]

    train_frame, train_keyframe_idx, train_label, train_caption = [], [], [], []
        
    n_episode = len(pkl_files)
    for eps_id in tqdm(range(n_episode)):
        pkl_file = os.path.join(current_folder_path, pkl_files[eps_id])
        video_frame, keyframe_idx, label, caption = data_preprocess(file_path=pkl_file)
        
        train_frame.append(video_frame)
        train_keyframe_idx.append(keyframe_idx)
        train_label.append(label)
        train_caption.append(caption) 
    
    # save dataset
    train_save_path = os.path.join(os.path.dirname(os.path.dirname(current_folder_path)), 'train_dataset.h5')
    with h5py.File(train_save_path, "w") as hf:
        group_1 = hf.create_group("video_frame")
        group_2 = hf.create_group("key_frame")
        group_3 = hf.create_group("label")
        group_4 = hf.create_group("caption")

        for i, arr in enumerate(train_frame):
            group_1.create_dataset(f"array_{i}", data=arr)

        for i, arr in enumerate(train_keyframe_idx):
            group_2.create_dataset(f"array_{i}", data=arr)
        
        for i, arr in enumerate(train_label):
            group_3.create_dataset(f"array_{i}", data=arr
                                   )
        for i, arr in enumerate(train_caption):
            group_4.create_dataset(f"array_{i}", data=arr)
    

def construct_test_dataset(current_folder_path, train_ratio=0.8):
    pkl_files = [file for file in os.listdir(current_folder_path) if file.endswith('.pkl')]

    test_frame, test_keyframe_idx, test_label, test_caption = [], [], [], []
        
    n_episode = len(pkl_files)
    test_count = n_episode - int(n_episode * train_ratio)
    for eps_id in tqdm(range(n_episode)):
        pkl_file = os.path.join(current_folder_path, pkl_files[eps_id])
        video_frame, keyframe_idx, label, caption = data_preprocess(file_path=pkl_file)
        
        if eps_id < test_count:
            test_frame.append(video_frame)
            test_keyframe_idx.append(keyframe_idx)
            test_label.append(label)
            test_caption.append(caption)
        else:
            break
    
    # save dataset
    test_save_path = os.path.join(os.path.dirname(os.path.dirname(current_folder_path)), 'test_dataset.h5')
    with h5py.File(test_save_path, "w") as hf:
        group_1 = hf.create_group("video_frame")
        group_2 = hf.create_group("key_frame")
        group_3 = hf.create_group("label")
        group_4 = hf.create_group("caption")

        for i, arr in enumerate(test_frame):
            group_1.create_dataset(f"array_{i}", data=arr)

        for i, arr in enumerate(test_keyframe_idx):
            group_2.create_dataset(f"array_{i}", data=arr)
        
        for i, arr in enumerate(test_label):
            group_3.create_dataset(f"array_{i}", data=arr)

        for i, arr in enumerate(test_caption):
            group_4.create_dataset(f"array_{i}", data=arr)


def construct_dataset(current_folder_path, train_ratio=0.8):
    pkl_files = [file for file in os.listdir(current_folder_path) if file.endswith('.pkl')]

    train_frame, train_keyframe_idx, train_label, train_caption = [], [], [], []
    test_frame, test_keyframe_idx, test_label, test_caption = [], [], [], []
        
    n_episode = len(pkl_files)
    train_count = int(n_episode * train_ratio)

    for eps_id in tqdm(range(n_episode)):
        pkl_file = os.path.join(current_folder_path, pkl_files[eps_id])
        video_frame, keyframe_idx, label, caption = data_preprocess(file_path=pkl_file)
        
        if eps_id < train_count:
            train_frame.append(video_frame)
            train_keyframe_idx.append(keyframe_idx)
            train_label.append(label)
            train_caption.append(caption) 
        else:
            test_frame.append(video_frame)
            test_keyframe_idx.append(keyframe_idx)
            test_label.append(label)
            test_caption.append(caption)
    
    # save dataset
    train_save_path = os.path.join(os.path.dirname(os.path.dirname(current_folder_path)), 'train_dataset.h5')
    test_save_path = os.path.join(os.path.dirname(os.path.dirname(current_folder_path)), 'test_dataset.h5')

    with h5py.File(train_save_path, "w") as hf:
        group_1 = hf.create_group("video_frame")
        group_2 = hf.create_group("key_frame")
        group_3 = hf.create_group("label")
        group_4 = hf.create_group("caption")

        for i, arr in enumerate(train_frame):
            group_1.create_dataset(f"array_{i}", data=arr)

        for i, arr in enumerate(train_keyframe_idx):
            group_2.create_dataset(f"array_{i}", data=arr)
        
        for i, arr in enumerate(train_label):
            group_3.create_dataset(f"array_{i}", data=arr
                                   )
        for i, arr in enumerate(train_caption):
            group_4.create_dataset(f"array_{i}", data=arr)
    with h5py.File(test_save_path, "w") as hf:
        group_1 = hf.create_group("video_frame")
        group_2 = hf.create_group("key_frame")
        group_3 = hf.create_group("label")
        group_4 = hf.create_group("caption")

        for i, arr in enumerate(test_frame):
            group_1.create_dataset(f"array_{i}", data=arr)

        for i, arr in enumerate(test_keyframe_idx):
            group_2.create_dataset(f"array_{i}", data=arr)
        
        for i, arr in enumerate(test_label):
            group_3.create_dataset(f"array_{i}", data=arr)

        for i, arr in enumerate(test_caption):
            group_4.create_dataset(f"array_{i}", data=arr)

def static_data(current_folder_path):
    pkl_files = [file for file in os.listdir(current_folder_path) if file.endswith('.pkl')]

    # task number
    task_list, task_length = [], []
    for pkl_file in tqdm(pkl_files):
        pkl_file_path = os.path.join(current_folder_path, pkl_file)
        with open(pkl_file_path, 'rb') as file:
            data = pickle.load(file)
        video = data['video'] 
        label = np.array(data['skill_index'], dtype=np.uint8)
        all_task = data['skill_name']
        task = all_task[:data['finished_task_number']]
        task_length.append(len(task))
        task_list.extend(task)
    task_set = set(task_list)
    task_dict = {task:task_list.count(task) for task in task_set}
    task_dict = dict(sorted(task_dict.items(), key=lambda x: x[1], reverse=False))
    task_num_dict = Counter(task_length)
    print(task_dict)
    print(task_num_dict)

def statci_video_length(current_folder_path):
    pkl_files = [file for file in os.listdir(current_folder_path) if file.endswith('.pkl')]

    # task number
    video_length = []
    for pkl_file in tqdm(pkl_files):
        pkl_file_path = os.path.join(current_folder_path, pkl_file)
        with open(pkl_file_path, 'rb') as file:
            data = pickle.load(file)
        video = data['video'] 
        video_length.append(video.shape[0])
    mean_video_length = np.mean(video_length)
    print(mean_video_length)

if __name__ == '__main__':
    file_path = "/data/ubuntu/VideoRLCS/dataset/calvin/dataset_100/"
    test_path = "/data/ubuntu/VideoRLCS/dataset/calvin/eval_log_v3/"
    # # static analysis
    # static_data(test_path)

    # statci_video_length(file_path)

    # data preprocess
    # construct_train_dataset(current_folder_path=file_path)
    # construct_test_dataset(current_folder_path=test_path)
    construct_dataset(current_folder_path=test_path)

    
