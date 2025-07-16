import cv2
import numpy as np
import h5py

from collections import Counter
from tqdm import tqdm

def read_dataset(dataset_file):
    with h5py.File(dataset_file, "r") as hf:
        if 'maniskill2' in dataset_file:
            group_1 = hf["base_camera"]
        else:
            group_1 = hf["video_frame"]
        # video_frames = [group_1[f"array_{i}"][:] for i in range(len(group_1))]
        
        # video frames resize to (224,224,3)
        target_size = (224, 224)
        video_frames = []
        for i in range(len(group_1)):
            video = group_1[f"array_{i}"][:]
            if video.shape[1] != 224 and video.shape[2] != 224:
                resized_video = []
                for frame in video:
                    resized_frame = cv2.resize(frame, target_size)
                    resized_video.append(resized_frame)
                resized_video = np.array(resized_video)
                video_frames.append(resized_video)
            else:
                video_frames.append(video)

        group_2 = hf["label"] 
        labels = [group_2[f"array_{i}"][:] for i in range(len(group_2))]

        group_3 = hf["caption"] 
        captions = [group_3[f"array_{i}"][:].astype('str') for i in range(len(group_3))]

        group_4 = hf["gtscore"] 
        gtscores = [group_4[f"array_{i}"][:] for i in range(len(group_4))]

        group_5 = hf["key_frame"]
        keyframe_indices = [group_5[f"array_{i}"][:] for i in range(len(group_5))]
    
    return video_frames, labels, captions, gtscores, keyframe_indices

def merge_dataset(dataset_files, target_file):
    video_list, label_list, caption_list, gtscore_list, keyframe_indices_list = [], [], [], [], []
    for dataset_file in dataset_files:
        print(dataset_file)
        video_frames, labels, captions, gtscores, keyframe_indices = read_dataset(dataset_file)
        video_list.extend(video_frames)
        label_list.extend(labels)
        caption_list.extend(captions)
        gtscore_list.extend(gtscores)
        keyframe_indices_list.extend(keyframe_indices)

    print(f"dataset saving: {target_file}")
    with h5py.File(target_file, "w") as hf:
        group_1 = hf.create_group("video_frame")
        group_2 = hf.create_group("label")
        group_3 = hf.create_group("caption")
        group_4 = hf.create_group("gtscore")
        group_5 = hf.create_group("key_frame")

        for i, arr in enumerate(video_list):
            group_1.create_dataset(f"array_{i}", data=arr)

        for i, arr in enumerate(label_list):
            group_2.create_dataset(f"array_{i}", data=arr)
        
        for i, arr in enumerate(caption_list):
            arr_bytes = [s.encode('utf-8') for s in arr]
            group_3.create_dataset(f"array_{i}", data=arr_bytes)
            
        for i, arr in enumerate(gtscore_list):
            group_4.create_dataset(f"array_{i}", data=arr)

        for i, arr in enumerate(keyframe_indices_list):
            group_5.create_dataset(f"array_{i}", data=arr)

if __name__ == '__main__':
    calvin_train_path = "/data/ubuntu/VideoRLCS/dataset/calvin/test_dataset2.h5"
    maniskill_train_file = "/data/ubuntu/VideoRLCS/dataset/maniskill2/train_dataset.h5"
    kitchen_train_file = "/data/ubuntu/VideoRLCS/dataset/kitchen/train_dataset.h5"

    calvin_test_path = "/data/ubuntu/VideoRLCS/dataset/calvin/test_dataset2.h5"
    maniskill_test_file = "/data/ubuntu/VideoRLCS/dataset/maniskill2/test_dataset.h5"
    kitchen_test_file = "/data/ubuntu/VideoRLCS/dataset/kitchen/test_dataset.h5"

    train_data_files = [calvin_train_path, maniskill_train_file, kitchen_train_file]
    test_data_files = [calvin_test_path, maniskill_test_file, kitchen_test_file]
    
    target_train_file = "/data/ubuntu/VideoRLCS/dataset/rebuttle/train_dataset.h5"
    target_test_file = "/data/ubuntu/VideoRLCS/dataset/rebuttle/test_dataset.h5"

    print("merget training dataset")
    merge_dataset(train_data_files, target_train_file)
    print("merge test dataset")
    merge_dataset(test_data_files, target_test_file)
    

    
