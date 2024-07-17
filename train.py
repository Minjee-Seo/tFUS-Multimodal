import os
import sys
import ast
import time
import datetime
import argparse
import torch
from torch.optim.lr_scheduler import LambdaLR

import warnings
warnings.filterwarnings("ignore")

from loader import load_dataset
from models_cnn import CNNModel, weights_init_cnn
from models_swin import SwinTransformer, weights_init_swin
from trainer import train_one_epoch, val_one_epoch
from utils import *

def arg_list(s):
    v = ast.literal_eval(s)
    return v

parser = argparse.ArgumentParser()
parser.add_argument("--num_epoch", type=int, default=100, help="training epoch for constant lr")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch for lr decay, set to 0 if not using learning rate scheduler")
parser.add_argument("--train_bs", type=int, default=8, help="batch size for training")
parser.add_argument("--valid_bs", type=int, default=8, help="batch size for validation")
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="Adam coefficient")
parser.add_argument("--b2", type=float, default=0.999, help="Adam coefficient")
parser.add_argument("--weight_decay", type=float, default=0.05, help="Regularization")
parser.add_argument('--modality', type=str, default='mri', help="Medical image modality (CT/MR)")
parser.add_argument('--run_name', type=str, default='test_run', help="Run name")
parser.add_argument('--model', type=str, default='cnn', help="Choose model to train : [ae/unet/swin]")
parser.add_argument('--save_best_model', action='store_true', default=True)
parser.add_argument('--init_model', action='store_true', default=False)
parser.add_argument('--cuda', action='store_true', default=False, help="Use GPU operation")
parser.add_argument('--ckpt', type=arg_list, default=None, help="Save checkpoint at certain epoch. usage: [1,10,100,150]")
opt = parser.parse_args()

def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch - opt.num_epoch) / float(opt.decay_epoch)
        return lr_l

if __name__ == "__main__":
    
    print(opt)

    os.makedirs("%s" % opt.run_name, exist_ok=True)

    # Define dataset and device
    train_dataloader, valid_dataloader = load_dataset(opt.modality, opt.train_bs, opt.valid_bs)
    train_epoch = opt.num_epoch + opt.decay_epoch
    device = torch.device('cuda') if opt.cuda else 'cpu'

    # Define model: load your saved state dict if needed
    if opt.model == 'ae': model = CNNModel(skip=False)
    elif opt.model=='unet': model=CNNModel(skip=True)
    elif opt.model == 'swin': model = SwinTransformer(device=device)
    
    if opt.init_model:
        model.apply(weights_init_swin) if opt.model=='swin' else model.apply(weights_init_swin)
    
    model = model.to(device)
    
    # Define optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda = lambda_rule) if opt.decay_epoch!=0 else None

    val_score_max = 0
    
    # Training process
    for epoch in range(1, train_epoch+1):
        
        tic = time.time()
         
        sys.stdout.write("\nTraining epoch %d/%d\n"%(epoch, train_epoch)) 
        model.train()
        train_one_epoch(model, optimizer, train_dataloader, device, scheduler)
        sys.stdout.write("\nValidation epoch %d/%d\n"%(epoch, train_epoch)) 
        model.eval()
        val_score = val_one_epoch(model, valid_dataloader, device)

        toc = time.time()

        time_remain = datetime.timedelta(seconds=(train_epoch+1-epoch)*(toc-tic))
        sys.stdout.write("\nETA : %s"%(time_remain))
        
        if opt.save_best_model and val_score > val_score_max:
            val_score_max = val_score
            torch.save(model.state_dict(), "%s/best_model.pt")

        if opt.ckpt!=None and epoch in opt.ckpt:
            torch.save(
                {'epoch': epoch,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 },
                "%s/epoch_%d.pt"%(opt.run_name, epoch)
            )

    # save the trained model
    torch.save(model.state_dict(), "%s/epoch_%d.pth"%(opt.run_name, train_epoch))