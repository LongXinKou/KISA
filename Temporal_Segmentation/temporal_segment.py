import numpy as np

from sklearn.cluster import SpectralClustering
from sklearn import metrics

from .cpd_auto import cpd_auto


def kts_segment(vFeature_array, m):
    '''
    input:
        vFeature_array:array(t,d)
        m:int
    output:
        cps:np.array(m)
        y_pred:np.array(t)
    '''
    K = np.dot(vFeature_array, vFeature_array.T)
    # cps, scores = cpd_nonlin(K, m, lmin=1, lmax=50)
    cps, scores = cpd_auto(K, ncp=vFeature_array.shape[0]-1, vmax=1)
    if (vFeature_array.shape[0]-1) not in cps:
        cps = np.append(cps, vFeature_array.shape[0]-1)

    return cps

def cluster_segment(vFeature_array, m):
    '''
    input:
        vFeature_array:array(t,d)
        m:int
    output:
        cps:np.array(m)
        y_pred:np.array(t)
    '''
    # clustering --> proposal
    y_pred = []
    pred_metrics = []
    for index, gamma in enumerate((0.01, 0.1, 1)):
        y_pred.append(SpectralClustering(n_clusters=m, gamma=gamma).fit_predict(vFeature_array))
        pred_metrics.append(metrics.calinski_harabasz_score(vFeature_array, y_pred[index]))
    index = np.argmax(pred_metrics)
    y_pred = y_pred[index]

    # select change point
    cps = []
    for i in range(m):
        cps.append(find_last_occurrence(y_pred, i))
    cps = np.sort(cps)
    for i in range(m):
        if i == 0:
            y_pred[:cps[i]+1] = i
        else:
            y_pred[cps[i-1]+1:cps[i]+1] = i
    return cps, y_pred
    

def find_last_occurrence(arr, target):
    '''
    input:
        arr:np.array
    '''
    last_occurrence = arr.shape[0] - np.where(arr[::-1]==target)[0][0] - 1
    return last_occurrence


def segment(vFeature_array, m, mode='kts'):
    '''
    input:
        mode:kts/cluster
    '''
    if mode == 'kts':
        return kts_segment(vFeature_array, m)
    elif mode == 'cluster':
        return cluster_segment(vFeature_array, m)