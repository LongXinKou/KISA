import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib import animation

def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out

def make_dir(args):
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)


def tensor_to_np(x):
    return x.data.cpu().numpy()

class Logger(object):
    """docstring for Logger."""

    def __init__(self, args):
        super(Logger, self).__init__()
        self.log_file = args.log_file
        _LOG_FORMAT = "%(asctime)s - %(levelname)s"
        logging.basicConfig(filename=args.log_file, level=logging.INFO, format=_LOG_FORMAT)
    def log(self,msg):
        logging.info(msg)



def save_checkpoint(model_state, optim_state, is_best, step, args, name=''):
    if(step % args.save_model_every_n_steps == 0):
        print("=> saving checkpoint '{}'".format(step))
        checkpoint = {
            'epoch': step,
            'model_state_dict': model_state,
            'optimizer_state_dict': optim_state,
            'params': {
                "hidden_size": args.hidden_size,
                "tfm_heads": args.tfm_heads,
                "tfm_layers": args.tfm_layers
            }
        }
        torch.save(checkpoint, os.path.join(args.save_dir, name + 'checkpoint_%03d.pth.tar' % step))
    if is_best:
        print("=> saving best checkpoint '{}'".format(step))
        checkpoint = {
            'epoch': step,
            'model_state_dict': model_state,
            'optimizer_state_dict': optim_state,
            'params': {
                "hidden_size": args.hidden_size,
                "tfm_heads": args.tfm_heads,
                "tfm_layers": args.tfm_layers
            }
        }
        torch.save(checkpoint, os.path.join(args.save_dir, name + 'model_best_epochs.pth.tar'))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def display_frames_as_gif(frames,dir='./test.gif'):
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=5)
    anim.save(dir, writer='imagemagick', fps=5)
    plt.close()

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):
    """Detect peaks in data based on their amplitude and other features.
    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).
    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.
    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's
    See this IPython Notebook [1]_.
    References
    ----------
    "Marcos Duarte, https://github.com/demotu/BMC"
    [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indexes of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indexes by their occurrence
        ind = np.sort(ind[~idel])
    
    return ind

def display_ConfidenceScore_as_png(x, y, png_save_dir=None, peak=None, label=None, color=None):
    color_keys = np.array([key for key,value in color.items()])
    plt.plot(x, y)
    plt.xlabel('Time Step')
    plt.ylabel('Confidence Score')
    plt.grid()
    if peak is not None:
        for i in range(np.max(label)+1):
            tmp = peak[i==label]
            plt.plot(tmp+1, y[tmp], "o", color=color_keys[i])
    plt.savefig(png_save_dir)
    plt.close()

def display_similarity_as_png(y,png_save_dir="./test.png",subgoal_index=None):
    '''
    input:
        subgoal_index:
    '''
    x = np.arange(y.shape[0])
    plt.plot(x, y)
    plt.xlabel('Time Step')
    plt.ylabel('Similarity Score')
    plt.grid()
    # ground truth subgoal
    if subgoal_index is not None:
        for subgoal in subgoal_index:
            plt.axvline(x=subgoal, color='r', linestyle='--')

    plt.savefig(png_save_dir)
    plt.close()

def display_similarity(y_pred, gtscore, png_save_dir="./test.png", subgoal_index=None,):
    '''
    input:
        y_pred:array(t,1)
        gtscore:array(t,1)
    '''
    x = np.arange(y_pred.shape[0])
    plt.plot(x, y_pred, label='pred', color='#AA5529', linestyle='-')
    plt.plot(x, gtscore, label='gtscore', color='#F5A95E', linestyle='--')

    plt.title('Similarity Comparasion') 
    plt.xlabel('Time Step')
    plt.ylabel('Similarity Score')
    plt.grid()
    plt.legend()
    
    # ground truth subgoal
    if subgoal_index is not None:
        for subgoal in subgoal_index:
            plt.axvline(x=subgoal, color='r', linestyle='--')

    plt.savefig(png_save_dir)
    plt.close()


def Calculate_Similarity(vFeature:torch.Tensor, tFeature:torch.Tensor, length=None, mode='cos'):
    '''
    input:
        vFeature:(b,t,c)
        tFeature:(b,t,c)
        length:(b,1)
    output:
        logits:(b,t,1)
    '''
    b,t,c = vFeature.shape
    if mode == 'cos':
        cosine_sim = F.cosine_similarity(vFeature, tFeature, dim=-1) #tensor(b,t)
        logits = torch.unsqueeze(cosine_sim, dim=-1) #tensor(b,t,1)
    return logits

def Calculate_Constrastive_Loss(vFeature, tFeature, captionFeature, tau=1.0):
    '''
    input:
        vFeature: (b,t,c) - video feature
        tFeature: (b,t,c) - text feature
        captionFeature: (k,c) - caption features for negative samples
        hFeature: (b,t,c) - historical frame features
    output:
        loss
    '''
    # Original positive pair
    numerator = Calculate_Similarity(vFeature, tFeature) 
    numerator = torch.exp(numerator / tau) #tensor(b,t,1)

    # 1. Incorrect Skill Alignments
    denoninator = []
    for cFeature in captionFeature:
        cFeature = torch.unsqueeze(torch.unsqueeze(cFeature, 0), 0).repeat(vFeature.shape[0], vFeature.shape[1], 1) #tensor(b,t,c)
        denoninator.append(torch.exp(Calculate_Similarity(vFeature, cFeature) / tau)) #tensor(b,t,1)
    denoninator = torch.stack(denoninator, dim=2) #tensor(b,t,k,1)
    denoninator = torch.sum(denoninator, dim=2) #tensor(b,t,1)

    # total negtive
    denoninator = denoninator - numerator 
    loss = -torch.log(numerator / denoninator) #tensor(b,t,1)
    loss = torch.mean(torch.mean(loss, dim=1), dim=0)
    
    return loss

def Calculate_Contrastive_Loss_Extended(vFeature, tFeature, captionFeature, tau=1.0):
    '''
    input:
        vFeature: (b,t,c) - video feature
        tFeature: (b,t,c) - text feature
        captionFeature: (k,c) - caption features for negative samples
    output:
        loss
    '''
    # Original positive pair
    numerator = Calculate_Similarity(vFeature, tFeature) 
    numerator = torch.exp(numerator / tau) #tensor(b,t,1)

    # 1. Incorrect Skill Alignments
    neg_sim_1 = []
    for cFeature in captionFeature:
        cFeature = cFeature.unsqueeze(0).unsqueeze(0).repeat(vFeature.shape[0], vFeature.shape[1], 1)
        neg_sim_1.append(torch.exp(Calculate_Similarity(vFeature, cFeature) / tau))
    neg_sim_1 = torch.stack(neg_sim_1, dim=2)

    # 2. Disjoint Frame History Compositions
    # Instead of using external historical frames, we'll use shuffled current frames
    shuffled_indices = torch.randperm(vFeature.shape[0])
    shuffled_vFeature = vFeature[shuffled_indices]
    neg_sim_2 = torch.exp(Calculate_Similarity(vFeature, shuffled_vFeature) / tau)

    # 3. Semantic Reversals via Video Inversion
    reversed_vFeature = torch.flip(vFeature, [1])  # Reverse the time dimension
    neg_sim_3 = torch.exp(Calculate_Similarity(reversed_vFeature, tFeature) / tau)

    # Combine all negative similarities
    total_neg_sim = torch.sum(neg_sim_1, dim=2) + neg_sim_2 + neg_sim_3

    loss = -torch.log(numerator / (numerator + total_neg_sim))
    loss = torch.mean(loss)

    return loss


def Calculate_gtscore_Loss(vFeature, tFeature, gtscore):
    pred_score = Calculate_Similarity(vFeature, tFeature)
    criterion = nn.MSELoss().to(pred_score.device)
    loss = criterion(pred_score, gtscore)
    return loss

def cross_entropy_loss(logits, target):
    '''
    input:
        logits:(b,t,n)
        target:(b,t)
    '''
    criterion = nn.CrossEntropyLoss().to(logits.device)
    logits = logits.view(-1, logits.shape[2]) #tensor(b*t,n)
    target = target.view(-1) #tensor(b*t)
    loss = criterion(logits, target)
    return loss

# Reconstruction + KL divergence losses summed over all elements and batch
def cvae_loss(reconstructed_data, target_data, mu, logvar, criterion):
    BCE = criterion(reconstructed_data, target_data)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def Calculate_F1_Score(prediction, target, adjacent=2):
    '''
    input:
        prediction:tensor(n)
        target:tensor(k)
    output:
        f1_score:tensor(1)
    '''
    TP, FP, FN = 0, 0, 0

    if prediction.numel() != 0:
        for pred in prediction:
            if adjacent == 1:
                if pred in target or pred - 1 in target or pred + 1 in target:
                    TP += 1
                else:
                    FP += 1
            elif adjacent == 2:
                if pred in target or (pred - 1 in target) or (pred - 2 in target) or (pred + 1 in target) or (pred + 2 in target):
                    TP += 1
                else:
                    FP += 1
    
    for true_label in target:
        if adjacent == 1:
            if true_label not in prediction and true_label - 1 not in prediction and true_label + 1 not in prediction:
                FN += 1
        elif adjacent == 2:
            if true_label not in prediction and (true_label - 1 not in prediction) and (true_label - 2 not in prediction) and (true_label + 1 not in prediction) and (true_label + 2 not in prediction):
                FN += 1
        
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return f1_score, precision, recall

def Calculate_MAE(prediction, target):
    '''
    input:
        prediction:tensor(n)
        target:tensor(k)
    output:
        mae:tensor(1)
    '''
    if prediction.size(0) != target.size(0):
        prediction_tmp = prediction.view(-1, 1).expand(-1, target.size(0))
    else:
        prediction_tmp = prediction
    absolute_errors = torch.abs(prediction_tmp - target)
    mae = absolute_errors.float().mean().item()
    return mae

def Calculate_FAR(prediction, target, video_length, adjacent=2):
    '''
    input:
        prediction:tensor(n)
        target:tensor(k)
    output:
        False Alarm Rate(1)
    '''
    TP, FP, FN = 0, 0, 0

    if prediction.numel() != 0:
        for pred in prediction:
            if adjacent == 1:
                if pred in target or pred - 1 in target or pred + 1 in target:
                    TP += 1
                else:
                    FP += 1
            elif adjacent == 2:
                if pred in target or (pred - 1 in target) or (pred - 2 in target) or (pred + 1 in target) or (pred + 2 in target):
                    TP += 1
                else:
                    FP += 1

    for true_label in target:
        if adjacent == 1:
            if true_label not in prediction and true_label - 1 not in prediction and true_label + 1 not in prediction:
                FN += 1
        elif adjacent == 2:
            if true_label not in prediction and (true_label - 1 not in prediction) and (true_label - 2 not in prediction) and (true_label + 1 not in prediction) and (true_label + 2 not in prediction):
                FN += 1

    false_alarm_rate = (FP+FN) / (video_length-target.shape[0])

    return false_alarm_rate

def Calculate_NACC(prediction, target):
    '''
    input:
        prediction:tensor(n)
        target:tensor(k)
    output:
        number_acc
    '''
    gt_number = target.shape[0]
    pred_number = prediction.shape[0]
    number_acc = abs(gt_number-pred_number) / gt_number
    return number_acc

def Calculate_Skill_F1_Score(prediction, target, num_classes):
    '''
    input:
        prediction:tensor(1,t)
        target:tensor(1,t)
        num_classes:list of class labels
    output:
        macro_f1_score:tensor(1)
    '''
    confusion_matrix = torch.zeros(len(num_classes), len(num_classes))
    for i in range(len(num_classes)):
        for j in range(len(num_classes)):
            confusion_matrix[i, j] = torch.sum((prediction == num_classes[j]) & (target == num_classes[i]))

    precision = torch.zeros(len(num_classes))
    recall = torch.zeros(len(num_classes))
    for i in range(len(num_classes)):
        true_positives = confusion_matrix[i, i]
        false_positives = torch.sum(confusion_matrix[:, i]) - true_positives
        false_negatives = torch.sum(confusion_matrix[i, :]) - true_positives

        precision[i] = true_positives / (true_positives + false_positives + 1e-9) 
        recall[i] = true_positives / (true_positives + false_negatives + 1e-9)

    f1_score_per_class = 2 * (precision * recall) / (precision + recall + 1e-9)
    macro_f1_score = torch.mean(f1_score_per_class)
    macro_precision = torch.mean(precision)
    macro_recall = torch.mean(recall)
    return macro_f1_score, macro_precision, macro_recall

def Calculate_Skill_IoU(tensor1, tensor2, num_classes):
    '''
    input:
        prediction:tensor(1,t)
        target:tensor(1,t)
        num_classes:list of class labels
    output:
        average_iou
    '''
    iou_sum = 0.0
    for idx in range(len(num_classes)):
        class_idx = num_classes[idx]
        mask1 = (tensor1 == class_idx).float()
        mask2 = (tensor2 == class_idx).float()

        intersection = torch.sum(mask1 * mask2)
        union = torch.sum(mask1 + mask2) - intersection

        class_iou = intersection / union if union != 0 else 0
        iou_sum += class_iou.item()

    average_iou = iou_sum / len(num_classes)
    return average_iou

def Calculate_Skill_TOPK(prediction, label, k=1):
    '''
    input:
        prediction:tensor(1,t,n)
        target:tensor(1,t)
        num_classes:list of class labels
    output:
        number_acc
    '''
    _, top_k_preds = torch.topk(prediction, k=k, dim=-1)

    correct_predictions = torch.any(top_k_preds == label.unsqueeze(-1), dim=-1)

    top_k_acc = torch.mean(correct_predictions.float())
    return top_k_acc.item()


def load_data(dataset_file, seed=100):
    with h5py.File(dataset_file, "r") as hf:
        group_1 = hf["video_frame"]
        base_camera_observations = [group_1[f"array_{i}"][:] for i in range(len(group_1))]

        group_4 = hf["label"] 
        labels = [group_4[f"array_{i}"][:] for i in range(len(group_4))]

        group_5 = hf["caption"] 
        captions = [group_5[f"array_{i}"][:].astype('str') for i in range(len(group_5))]
    
    dataset_len = len(base_camera_observations)
    seed = random.randint(0, dataset_len-1)

    base_observations = base_camera_observations[seed] #array(t,h,w,3)
    label = labels[seed] #array(t,1)
    caption = captions[seed] #list(t)

    base_observation = np.transpose(base_observations,[0,3,1,2])
    base_observation = base_observation / 255.
    base_observation = torch.tensor(base_observation).float() #tensor(t,3,h,w)
    label = torch.tensor(label) #tensor(t,1)
    
    base_observation = torch.unsqueeze(base_observation,dim=0) #tensor(1,t,3,h,w)
    label = torch.unsqueeze(label,dim=0) #tensor(1,t,1)
    caption = [caption] #list(1,t)
    
    return base_observation, label, caption