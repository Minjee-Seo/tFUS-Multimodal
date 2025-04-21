# helpers
import os
import sys
import torch
from datetime import timedelta
from time import time
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
    total_dice = 0
    total_dist = 0
    total_delta = 0
    batch_size = pred_list.size(0)
    
    for i in range(batch_size):
        pred = pred_list[i].detach()
        target = target_list[i].detach()
        
        dice = calculate_dice(pred, target)
        dist = calculate_dist(pred, target)
        delta = calculate_delta(pred, target)
        
        total_dice += dice
        total_dist += dist
        total_delta += delta
        
    return total_dice/batch_size, total_dist/batch_size, total_delta/batch_size

def train_one_epoch(epoch, train_epoch, model, optimizer, train_dataloader, device, scheduler=None):

    model.train()
    
    loss_tot = 0
    dice_tot = 0
    dist_tot = 0
    delta_tot = 0
    criterion = torch.nn.MSELoss().to(device)
    total_batches = train_epoch * (len(train_dataloader))
    prev_time = time()
    
    for i, batch in enumerate(train_dataloader):
        
        # Model inputs
        ff = batch["A"].unsqueeze(1).to(device)
        skull = batch["S"].unsqueeze(1).to(device)
        target = batch["B"].unsqueeze(1).to(device)
        tinput = batch["T"].to(device)
        
        scaler = MinMaxScaling(ff)
        ff = scaler.fit_transform(ff, ff=False)
        target = scaler.fit_transform(target, ff=True)

        optimizer.zero_grad()

        # compute loss
        pred = model(ff, skull, tinput)
        
        loss = criterion(pred, target)
        loss.mean().backward()
        loss_tot += loss.mean().item()     
        
        optimizer.step()
        
        # Calculate dice score
        train_dice, train_dist, train_delta = batch_metrics(pred, target)
        dice_tot += train_dice
        dist_tot += train_dist
        delta_tot += train_delta
        
        batches_done = (epoch - 1) * (len(train_dataloader)) + i + 1
        batches_left = total_batches - batches_done
        
        time_remain = timedelta(seconds=batches_left * (time() - prev_time))
        prev_time = time()

        sys.stdout.write(
            "\r[Epoch %d/%d] [Train batch %d/%d] [loss : %f] [Dice : %.2f] [Dist : %.2f] [Delta : %.2f] [Train ETA : %s]"
            % (
                epoch,
                train_epoch,
                i+1,
                len(train_dataloader),
                loss.mean().item(),
                train_dice*100,
                train_dist,
                train_delta*100,
                str(time_remain)[:-7]
            )
        )
        sys.stdout.flush()
        print('')
    
    if scheduler!=None: scheduler.step()
    
    return loss_tot/len(train_dataloader), dice_tot/len(train_dataloader), dist_tot/len(train_dataloader), delta_tot/len(train_dataloader)
    
def val_one_epoch(epoch, train_epoch, model, valid_dataloader, device):

    model.eval()
    loss_tot = 0
    dice_tot = 0
    dist_tot = 0
    delta_tot = 0
    criterion = torch.nn.MSELoss().to(device)
    
    total_batches = train_epoch * (len(valid_dataloader))
    prev_time = time()

    with torch.no_grad():
        
        for i, batch in enumerate(valid_dataloader):
            
            # Model inputs
            ff = batch["A"].unsqueeze(1).to(device)
            skull = batch["S"].unsqueeze(1).to(device)
            target = batch["B"].unsqueeze(1).to(device)
            tinput = batch["T"].to(device)

            scaler = MinMaxScaling(ff)
            ff = scaler.fit_transform(ff, ff=False)
            target = scaler.fit_transform(target, ff=True)

            # compute loss
            pred = model(ff, skull, tinput)
            loss = criterion(pred, target)
            loss_tot += loss.mean().item()     
            
            # Calculate dice score
            valid_dice, valid_dist, valid_delta = batch_metrics(pred, target)
            dice_tot += valid_dice
            dist_tot += valid_dist
            delta_tot += valid_delta
            
            batches_done = (epoch - 1) * (len(valid_dataloader)) + i + 1
            batches_left = total_batches - batches_done
            
            time_remain = timedelta(seconds=batches_left * (time() - prev_time))
            prev_time = time()

            sys.stdout.write(
                "\r[Epoch %d/%d] [Valid batch %d/%d] [loss : %f] [Dice : %.2f] [Dist : %.2f] [Delta : %.2f] [Val ETA : %s]"
                % (
                    epoch,
                    train_epoch,
                    i+1,
                    len(valid_dataloader),
                    loss.mean().item(),
                    valid_dice*100,
                    valid_dist,
                    valid_delta*100,
                    str(time_remain)[:-7]
                )
            )
            sys.stdout.flush()
    
    print('')
        
    return loss_tot/len(valid_dataloader), dice_tot/len(valid_dataloader), dist_tot/len(valid_dataloader), delta_tot/len(valid_dataloader)

def eval(model, test_dataloader, device):

    pred_list = []
    target_list = []
    
    total_dice = []
    total_dist = []
    total_delta = []
    
    model.eval()
    with torch.no_grad():
        with tqdm(test_dataloader, unit='batch') as testprogress:
            for batch in testprogress:

                # Model inputs
                ff = batch["A"].unsqueeze(1).to(device)
                skull = batch["S"].unsqueeze(1).to(device)
                target = batch["B"].unsqueeze(1)
                tinput = batch["T"].to(device)
                
                # Optional : apply MinMaxscaling to inputs
                scaler = MinMaxScaling(ff)
                ff = scaler.fit_transform(ff, ff=False)
                target = scaler.fit_transform(target, ff=True)

                # output prediction
                pred = model(ff, skull, tinput).cpu()
                dice, dist, delta = batch_metrics(pred, target)

                total_dice.append(dice)
                total_dist.append(dist)
                total_delta.append(delta)
                
                pred_list.append(pred.squeeze().detach())
                target_list.append(target.squeeze())

    # flattening result list
    target_list = [item for sublist in target_list for item in sublist]
    pred_list = [item for sublist in pred_list for item in sublist]

    return target_list, pred_list, total_dice, total_dist, total_delta
    
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