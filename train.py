import os
import ast
import time
import argparse
import torch
import pickle
from torch.optim.lr_scheduler import LambdaLR

import warnings
warnings.filterwarnings("ignore")

from dataset import load_dataset
from models_cnn import CNNModel, weights_init_cnn
from models_swin import SwinUNet, weights_init_swin
from utils import train_one_epoch, val_one_epoch

def arg_list(s):
    v = ast.literal_eval(s)
    return v

parser = argparse.ArgumentParser()
parser.add_argument("--num_epoch", type=int, default=100, help="train epoch for constant lr")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch for lr decay, set to 0 if not using learning rate scheduler")
parser.add_argument("--train_bs", type=int, default=8, help="batch size for training")
parser.add_argument("--valid_bs", type=int, default=8, help="batch size for validation")
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="Adam coefficient")
parser.add_argument("--b2", type=float, default=0.999, help="Adam coefficient")
parser.add_argument("--weight_decay", type=float, default=0.05, help="Regularization")
parser.add_argument('--modality', type=str, default='mri', help="Medical image modality (CT/MR)")
parser.add_argument('--run_name', type=str, default='test_run', help="Run name")
parser.add_argument('--model', type=str, default='unet', help="Choose model to train : [ae/unet/swin]")
parser.add_argument('--save_best_model', action='store_true', default=True)
parser.add_argument('--init_model', action='store_true', default=False)
parser.add_argument('--cuda', action='store_true', default=False, help="Use GPU")
parser.add_argument('--ckpt', type=arg_list, default=None, help="Save checkpoint at certain epoch. usage: [1,10,100,150]")
opt = parser.parse_args()

def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch - opt.num_epoch) / float(opt.decay_epoch)
        return lr_l

if __name__ == "__main__":
    
    print(opt)
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
    train_dice, train_dist, train_delta = [], [], []
    
    # Training process
    for epoch in range(1, train_epoch+1):
        
        start_time = time.time()
         
        model.train()
        dice, dist, delta = train_one_epoch(epoch, train_epoch, model, optimizer, train_dataloader, device, scheduler)
        model.eval()
        val_score = val_one_epoch(epoch, train_epoch, model, valid_dataloader, device)
        
        if opt.save_best_model and val_score > val_score_max:
            val_score_max = val_score
            torch.save(model.state_dict(), "%s/best_model.pth"%opt.run_name)

        if opt.ckpt!=None and epoch in opt.ckpt:
            torch.save(
                {'epoch': epoch,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 },
                "%s/epoch_%d.pth"%(opt.run_name, epoch)
            )
        
        train_dice.append(dice)
        train_dist.append(dist)
        train_delta.append(delta)
            
    train_results = {'dice':train_dice, 'dist':train_dist, 'delta':train_delta}
    
    # save the trained model and results
    torch.save(model.state_dict(), "%s/epoch_%d.pth"%(opt.run_name, train_epoch))
    
    with open('%s/epoch_%d.pickle','wb') as f:
        pickle.dump(train_results, f)
