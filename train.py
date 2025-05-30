import os
import ast
import time
import argparse
import torch
import pickle
import wandb
from torch.optim.lr_scheduler import LambdaLR

import warnings
warnings.filterwarnings("ignore")

from dataset import load_dataset
from models_cnn import CNNModel, weights_init_cnn
from models_swin import SwinUNet, weights_init_swin
from utils import train_one_epoch, val_one_epoch
from config import load_train_config

def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch - opt.num_epoch) / float(opt.decay_epoch)
        return lr_l

if __name__ == "__main__":
    
    opt = load_train_config()
    # print(opt)

    os.makedirs("%s" % opt.run_name, exist_ok=True)
    
    device = torch.device('cuda') if opt.cuda else 'cpu'

    # Define dataset
    train_dataloader, valid_dataloader = load_dataset(opt.modality, opt.train_bs, opt.valid_bs)
    train_epoch = opt.num_epoch + opt.decay_epoch

    # Define model: load your saved state dict if needed
    if opt.model == 'ae': model = CNNModel(skip=False)
    elif opt.model=='unet': model=CNNModel(skip=True)
    elif opt.model == 'swin': model = SwinUNet(device=device)
    
    if opt.init_model:
        model.apply(weights_init_swin) if opt.model=='swin' else model.apply(weights_init_cnn)
    
    model = model.to(device)
    
    # Define optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda = lambda_rule) if opt.decay_epoch!=0 else None

    val_score_max = 0
    train_loss, train_dice, train_dist, train_delta, valid_loss, valid_dice, valid_dist, valid_delta = [], [], [], [], [], [], [], []

    if opt.wandb_pj!=None:
        wandb.init(project=opt.wandb_pj, config=vars(opt))
        wandb.watch(model, log='all')
    
    # Training process
    for epoch in range(1, train_epoch+1):
        
        start_time = time.time()

        loss, dice, dist, delta = train_one_epoch(epoch, train_epoch, model, optimizer, train_dataloader, device, scheduler)
        v_loss, v_dice, v_dist, v_delta = val_one_epoch(epoch, train_epoch, model, valid_dataloader, device)
        
        if opt.save_best_model and v_dice > val_score_max:
            val_score_max = v_dice
            torch.save(model.state_dict(), "%s/best_model.pth"%opt.run_name)

        if opt.ckpt!=None and epoch in opt.ckpt:
            torch.save(
                {'epoch': epoch,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 },
                "%s/epoch_%d.pth"%(opt.run_name, epoch)
            )
        
        train_loss.append(loss)
        train_dice.append(dice)
        train_dist.append(dist)
        train_delta.append(delta)

        valid_loss.append(v_loss)
        valid_dice.append(v_dice)
        valid_dist.append(v_dist)
        valid_delta.append(v_delta)

        if opt.wandb_pj!=None:
            wandb.log({"train_loss":loss,
                        "train_dice":dice,
                        "train_dist":dist,
                        "train_delta":delta,
                        "valid_loss":v_loss,
                        "valid_dice":v_dice,
                        "valid_dist":v_dist,
                        "valid_delta":v_delta})
            
    train_results = {'train_loss':train_loss,
                     'train_dice':train_dice,
                     'train_dist':train_dist,
                     'train_delta':train_delta,
                     'valid_loss':valid_loss,
                     'valid_dice':valid_dice,
                     'valid_dist':valid_dist,
                     'valid_delta':valid_delta}
    
    # save the trained model and results
    torch.save(model.state_dict(), "%s/epoch_%d.pth"%(opt.run_name, train_epoch))
    
    with open('%s/epoch_%d.pickle'%(opt.run_name, train_epoch),'wb') as f:
        pickle.dump(train_results, f)

    if opt.wandb_pj!=None:
        wandb.finish()