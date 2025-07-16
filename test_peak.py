 # -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
import os
import json
import random
import h5py

from tqdm import tqdm
from utils import Calculate_Similarity, Calculate_F1_Score, Calculate_MAE, Calculate_FAR, Calculate_NACC, Calculate_Skill_IoU, Calculate_Skill_TOPK
from config import *
from Models import load_vlm
from Temporal_Segmentation.temporal_segment import *
from utils import detect_peaks

def load_data(dataset_file, seed=100):
    with h5py.File(dataset_file, "r") as hf:
        group_1 = hf["video_frame"]
        base_camera_observations = [group_1[f"array_{i}"][:] for i in range(len(group_1))]

        group_2 = hf["key_frame"]
        keyframe_indices = [group_2[f"array_{i}"][:] for i in range(len(group_2))]

        group_4 = hf["label"] 
        labels = [group_4[f"array_{i}"][:] for i in range(len(group_4))]

        group_4 = hf["caption"]
        captions = [group_4[f"array_{i}"][:].astype('str') for i in range(len(group_4))]
    
    dataset_len = len(base_camera_observations)
    seed = random.randint(0, dataset_len-1)

    base_observations = base_camera_observations[seed] #array(t,h,w,3)
    keyframe_idx = keyframe_indices[seed]
    label = labels[seed] #array(t,1)
    caption = captions[seed] #list(t)

    base_observation = np.transpose(base_observations,[0,3,1,2])
    base_observation = base_observation / 255.
    base_observation = torch.tensor(base_observation).float() #tensor(t,3,h,w)
    label = torch.tensor(label) #tensor(t,1)
    base_observation = torch.unsqueeze(base_observation,dim=0) #tensor(1,t,3,h,w)
    label = torch.unsqueeze(label,dim=0) #tensor(1,t,1)
    caption = [caption] #list(1,t)
    
    return base_observation, label, caption, keyframe_idx

def evaluate(args, model, skill_library, mpd=1):
    slFeature = model.get_text_feature(skill_library) #(n,c)

    f1_score_list, precision_list, recall_list = [], [], []
    mae_list, far_list, nacc_list = [], [], []
    iou_list = []
    t1_list, t3_list = [], []
    for seed in tqdm(range(100), desc='test'):
        # frame, label, caption = load_data(args.test_path, seed=seed)
        frame, label, caption, keyframe_idx = load_data(args.test_path, seed=seed)
        frame = frame.to(args.device) #tensor(bs,t,3,h,w)
        label = label.to(args.device) #tensor(bs,t)

        # get video feature and instruction feature
        vFeature, cFeature = model(frame, caption=caption) #tensor(b,t,c)/(b,t,c)
            
        # video-language retreival
        logits = []
        for Feature in slFeature:
            tFeature = torch.unsqueeze(torch.unsqueeze(Feature, 0), 0).repeat(1, vFeature.shape[1], 1) #tensor(1,t,c)
            logits.append(Calculate_Similarity(vFeature, tFeature)) #tensor(1,t,1)
        logits = torch.stack(logits, dim=2) #tensor(1,t,k,1)
        logits = torch.squeeze(logits, dim=-1)
        max_value, pred_idx = logits.max(dim=2) #tensor(1,t)/tensor(1,t)

        # keyframe index(last keyframe-->goal)
        true_keyframe_index = torch.tensor(keyframe_idx).to(args.device)
        flat_prediction = pred_idx.view(-1)
        flat_truth = label.view(-1)

        # v1: label keyframe detection
        pred_keyframe_index  = torch.squeeze((flat_prediction[:-1] != flat_prediction[1:]).nonzero())
        # v2: peak keyframe detection(label enchanced)
        confidence_score_array = torch.squeeze(max_value).detach().cpu().numpy()
        peak_index = detect_peaks(confidence_score_array, mpd=mpd)
        pred_keyframe_index3 = []
        for idx,val in enumerate(peak_index):
            if idx < peak_index.shape[0]-1:
                if flat_prediction[val] != flat_prediction[peak_index[idx+1]]:
                    pred_keyframe_index3.append(val)
                else:
                    continue
            else:
                pred_keyframe_index3.append(val)
        pred_keyframe_index = torch.tensor(pred_keyframe_index3).to(pred_keyframe_index.device)

        if pred_keyframe_index.dim() == 0 and pred_keyframe_index.numel() != 0: # dim
            pred_keyframe_index = torch.unsqueeze(pred_keyframe_index, dim=0)
        if flat_prediction[keyframe_idx[-1]] == flat_truth[keyframe_idx[-1]] and keyframe_idx[-1] not in pred_keyframe_index.tolist():
            goal_index = torch.tensor([keyframe_idx[-1]]).to(args.device)
            if pred_keyframe_index.numel() == 0: # empty
                pred_keyframe_index = goal_index

            pred_keyframe_index = torch.cat((pred_keyframe_index, goal_index))

        unique_values, unique_indices = torch.unique(pred_idx, return_inverse=True)

        # keyframe 
        # [metric-1] F1-score/Precision/Recall
        f1_score, precision, recall = Calculate_F1_Score(pred_keyframe_index, true_keyframe_index, adjacent=2)
        f1_score_list.append(f1_score)
        precision_list.append(precision)
        recall_list.append(recall)

        # [metric-2] MAE
        mae = Calculate_MAE(pred_keyframe_index, true_keyframe_index)
        if np.isnan(mae):
            mae = frame.shape[1]
        mae_list.append(mae)

        # [metric-3] False Alarm Rate
        far = Calculate_FAR(pred_keyframe_index, true_keyframe_index, video_length=frame.shape[1])
        far_list.append(far)

        # [metric-4]Number Acc
        number_acc = Calculate_NACC(pred_keyframe_index, true_keyframe_index)
        nacc_list.append(number_acc)

        # skill
        # [metric-1]IoU
        iou_score = Calculate_Skill_IoU(pred_idx, label, unique_values)
        iou_list.append(iou_score)

        # [metric-2]Top-K
        t1 = Calculate_Skill_TOPK(logits, label, k=1)
        t1_list.append(t1)
        t3 = Calculate_Skill_TOPK(logits, label, k=3)
        t3_list.append(t3)

    avg_nacc = np.mean(nacc_list)
    avg_f1_score = np.mean(f1_score_list)
    avg_mae = np.mean(mae_list)
    avg_far = np.mean(far_list)
    avg_iou = np.mean(iou_list)
    avg_t1 = np.mean(t1_list)
    avg_t3 = np.mean(t3_list)
    return avg_nacc, avg_f1_score, avg_mae, avg_far, avg_iou, avg_t1, avg_t3


if __name__ == '__main__':  
    args = test_config()

    library_path = os.path.dirname(args.test_path)
    with open(os.path.join(library_path,'skill.json'), 'r') as json_file:
        task_information = json.load(json_file)
    skill_library = task_information["skill"]
    args.library_size = len(skill_library)

    model = load_vlm(args, visual_representation=args.visual_representation, model_path=args.model_path, pretrain=True).to(args.device)
    model.eval()

    print("1. video text retrieval")
    seed_list = [42, 32, 2]
    nacc_list, f1_list, mae_list, far_list, iou_list, t1_list, t3_list = [], [], [], [], [], [], []
    for i in range(len(seed_list)):
        print(f'seed: {seed_list[i]}')
        np.random.seed(seed_list[i])
        random.seed(seed_list[i])

        avg_nacc, avg_f1_score, avg_mae, avg_far, avg_iou, avg_t1, avg_t3 = evaluate(args, model, skill_library, mpd=20)
        iou_list.append(avg_iou)
        t1_list.append(avg_t1)
        t3_list.append(avg_t3)

        nacc_list.append(avg_nacc)
        f1_list.append(avg_f1_score)
        mae_list.append(avg_mae)
        far_list.append(avg_far)
    

    avg_nacc = np.mean(nacc_list)
    var_nacc = np.std(nacc_list)
    print('Number Error:', avg_nacc)
    print('Number Error std:', var_nacc)
    avg_f1_score = np.mean(f1_list)
    var_f1_score = np.std(f1_list)
    print('F1_score:', avg_f1_score * 100)
    print('F1_score std:', var_f1_score * 100)
    avg_mae = np.mean(mae_list)
    var_mae = np.std(mae_list)
    print('MAE:', avg_mae)
    print('MAE std:', var_mae)
    avg_far = np.mean(far_list)
    var_far = np.std(far_list)
    print('Average Alarm False:', avg_far * 100)
    print('Average Alarm False std:', var_far * 100)
    print('===========skill matrics==============')
    avg_iou = np.mean(iou_list)
    var_iou = np.std(iou_list)
    print('IoU:', avg_iou * 100)
    print('IoU std:', var_iou * 100)
    avg_t1 = np.mean(t1_list)
    var_t1 = np.std(t1_list)
    print('Top-1 Accuracy:', avg_t1 * 100)
    print('Top-1 Accuracy std:', var_t1 * 100)
    avg_t3 = np.mean(t3_list)
    var_t3 = np.std(t3_list)
    print('Top-3 Accuracy:', avg_t3 * 100)
    print('Top-3 Accuracy std:', var_t3 * 100)