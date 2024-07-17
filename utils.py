# helpers
import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Tuple
import math
import matplotlib.pyplot as plt

class MinMaxScaling:
    '''
    input : pytorch tensor
    perform Min-Max scaling to batched input (expected size B*1*D*H*W)
    '''
    def __init__(self, X):
        self.min_value = X.view(X.size(0),-1).min(1)[0]
        self.max_value = X.view(X.size(0),-1).max(1)[0]
        
        self.ff_max_value = 479.1307
        self.ff_min_value = 1.4112
    
    def fit_transform(self, X, ff=True):
        X_std = X.clone()
        if ff:
            for i in range(len(X_std)):
                X_std[i] = (X[i] - self.ff_min_value) / (self.ff_max_value - self.ff_min_value)    
        
        else:
            for i in range(len(X_std)):
                X_std[i] = (X[i] - self.min_value[i]) / (self.max_value[i] - self.min_value[i])    
        
        return X_std
    
    def inverse_transform(self, X, ff=True):
        X_inv = X.clone()
        
        if ff:
            for i in range(len(X_inv)):
                X_inv[i] = X[i] * (self.ff_max_value - self.ff_min_value) + self.ff_min_value
            
        else:
            for i in range(len(X_inv)):
                X_inv[i] = X[i] * (self.max_value[i] - self.min_value[i]) + self.min_value[i]
        
        return X_inv

def unravel_indices(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    '''
    pytorch implementation of numpy function <unravel_indices>
    '''
    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord[::-1], dim=-1)

    return coord

def thr_tensor(ts):
    '''
    input : pytorch tensor
    for thresholding pytorch tensors
    '''
    max_value = ts.max()*0.5
    return (ts>=max_value).float()

def thr_npy(npy):
    '''
    input : numpy array
    for thresholding numpy arrays
    '''
    max_value = npy.max()*0.5
    return (npy>=max_value).astype(float)
        
def calculate_dice(pred, target):
    '''
    input : pytorch tensor
    computes dice score between two tensors
    '''
    max_pred = torch.max(pred)
    max_target = torch.max(target)
    
    pred = (pred.contiguous().view(-1)>=max_pred*0.5).float()
    target = (target.contiguous().view(-1)>=max_target*0.5).float()
    
    count_p = torch.count_nonzero(pred)
    count_t = torch.count_nonzero(target)
    count = torch.count_nonzero(((pred==1)&(target==1)).float())

    dice = count*2 / (count_p+count_t)
    
    return dice

def calculate_dist(pred, target):
    '''
    input : pytorch tensor
    calculates distance from maximum value points between two tensors
    output : [mm]
    '''
    idx_pred = unravel_indices(torch.argmax(pred), pred.shape)
    idx_target = unravel_indices(torch.argmax(target), target.shape)
    dist = math.sqrt(((idx_pred-idx_target)**2).sum(axis=0))*0.5
    
    return dist

def calculate_delta(pred, target):
    max_pred = torch.max(pred)
    max_target = torch.max(target)
    ratio = abs(max_pred - max_target)
    
    return ratio

def batch_metrics(pred_list, target_list):
    '''
    input : pytorch tensor
    computes Dice score while training and validation
    '''
    total_dice = 0
    total_dist = 0
    total_delta = 0
    batch_size = pred_list.size(0)
    
    for i in range(batch_size):
        pred = pred_list[i]
        target = target_list[i]
        
        dice = calculate_dice(pred, target)
        dist = calculate_dist(pred, target)
        delta = calculate_delta(pred, target)
        
        total_dice += dice
        total_dist += dist
        total_delta += delta
        
    return total_dice/batch_size, total_dist/batch_size, total_delta/batch_size
    
def savefig(results, pj_name, thres=False):
    
    def show_plane(ax, plane, cmap="jet", title=None):
        ax.imshow(plane, cmap=cmap)
        ax.axis("off")
        if title:
            ax.set_title(title)

    path = "./images/%s" % pj_name
    cmap = "jet"
    
    if thres:
        path = path + "/thsd"
        cmap = "viridis"
        
    os.makedirs(path, exist_ok=True)
    
    for i in tqdm(range(len(results))):
        output = np.array(results[i])
        if thres: output = thr_npy(output)
        _, (a1) = plt.subplots(ncols=1, figsize=(5,5))

        show_plane(a1, output[:, 56, :], cmap=cmap, title=f'Data_{i+1}')
        plt.savefig(path+f'/Data{i+1}.png', bbox_inches='tight',pad_inches=0.1)
        plt.cla()
        plt.clf()
        plt.close()
        
def create_split(num_patients, ratio):
    data_length = num_patients * 400
    sub_length = int(400 * ratio)

    test_indices = []

    for i in range(num_patients):
        start_idx = i * (data_length // num_patients)
        patient_indices = torch.arange(start_idx, start_idx + (data_length // num_patients))[:sub_length]
        test_indices.append(patient_indices)

    test_indices = torch.cat(test_indices)

    train_indices = torch.tensor([idx for idx in range(data_length) if idx not in test_indices])

    return train_indices, test_indices