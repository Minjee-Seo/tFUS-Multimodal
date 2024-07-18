import argparse
import numpy as np
import torch

import warnings
warnings.filterwarnings("ignore")

from dataset import load_dataset
from models_cnn import CNNModel
from models_swin import SwinUNet
from utils import eval, savefig

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', default=False, help="Use GPU")
parser.add_argument('--batch_size', type=int, default=8, help="Test batch size")
parser.add_argument('--modality', type=str, default='mri', help="Select med.image modality (CT/MR)")
parser.add_argument('--model', type=str, default='unet', help="Choose model to train : [ae/unet/swin]")
parser.add_argument('--run_name', type=str, default='test_run', help="project name to load")
parser.add_argument('--run_epoch', type=int, default=200, help="Trained epoch")
parser.add_argument('--plot', action='store_true', default=False, help="Save result images")
opt = parser.parse_args()

if __name__ == "__main__":

    PATH = "./%s/epoch_%d.pth" % (opt.run_name, opt.run_epoch)

    test_dataloader = load_dataset(opt.modality, test=True, test_batch_size=opt.batch_size)
    device = torch.device('cuda') if opt.cuda else 'cpu'

    # Define model
    if opt.model == 'ae': model = CNNModel(skip=False)
    elif opt.model=='unet': model=CNNModel(skip=True)
    elif opt.model == 'swin': model = SwinUNet(device=device)
    
    saved_dict = torch.load(PATH, map_location=device)
    model.load_state_dict(saved_dict['model_state_dict']) if 'model_state_dict' in saved_dict.keys() else model.load_state_dict(saved_dict)
    model.to(device)

    pred, target, dice, dist, delta = eval(model, test_dataloader, device)
    
    print("===============================")
    print("Dice : %.2f +- %.2f"%(np.mean(dice)*100, np.std(dice)*100))
    print("Dist : %.2f +- %.2f"%(np.mean(dist), np.std(dist)))
    print("Delta : %.2f +- %.2f"%(np.mean(delta)*100, np.std(delta)*100))
    
    # optional : save result images
    if opt.plot: savefig(target, opt.run_name, thres=False)