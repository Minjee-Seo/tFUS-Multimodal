import sys
import torch
from utils import *

def train_one_epoch(model, optimizer, train_dataloader, device, scheduler=None):
    
    loss_tot = 0
    dice_tot = 0
    criterion = torch.nn.MSELoss().to(device)

    for i, batch in enumerate(train_dataloader):
         
        # Model inputs
        ff = batch["A"].unsqueeze_(1).to(device)
        skull = batch["S"].unsqueeze_(1).to(device)
        target = batch["B"].unsqueeze_(1).to(device)
        tinput = batch["T"].to(device)
        
        # Optional : apply MinMaxscaling to input if not scaled
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
        train_dice = batch_metrics(pred, target)
        dice_tot += train_dice

        sys.stdout.write(
            "\r[Batch %d/%d] [loss : %f] [Dice : %f]"
            % (
                i+1,
                len(train_dataloader),
                loss.mean().item(),
                train_dice,
            )
        )
    
    if scheduler!=None: scheduler.step()
    
def val_one_epoch(model, valid_dataloader, device):

    loss_tot = 0
    dice_tot = 0
    dist_tot = 0
    delta_tot = 0
    criterion = torch.nn.MSELoss().to(device)

    for i, batch in enumerate(valid_dataloader):
         
        # Model inputs
        ff = batch["A"].unsqueeze(1).to(device)
        skull = batch["S"].unsqueeze(1).to(device)
        target = batch["B"].unsqueeze(1).to(device)
        tinput = batch["T"].to(device)
        
        # Optional : apply MinMaxscaling to inputs
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

        sys.stdout.write(
            "\r[Batch %d/%d] [loss : %f] [Dice : %f]"
            % (
                i+1,
                len(valid_dataloader),
                loss.mean().item(),
                valid_dice,
            )
        )
        
    return valid_dice