import sys
import time
import datetime
import argparse
import numpy as np
import torch

import warnings
warnings.filterwarnings("ignore")

from loader import load_dataset
from models_cnn import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', default=True, help="Use GPU operation")
parser.add_argument('--batch_size', type=int, default=8, help="Test batch size")
parser.add_argument('--modality', type=str, default='mri', help="Select med.image modality (CT/MR)")
parser.add_argument('--run_name', type=str, default='test_run', help="run name to load")
parser.add_argument('--run_epoch', type=int, default=200, help="saved model epoch")
parser.add_argument('--parallel', action='store_true', default=False)
opt = parser.parse_args()

def eval(model, test_dataloader):

    pred_list = []
    target_list = []
    dice_score = []
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            
            tic = time.time()
            
            # Model inputs
            ff = batch["A"].unsqueeze(1).to(device)
            skull = batch["S"].unsqueeze(1).to(device)
            target = batch["B"]
            tinput = batch["T"].to(device)
            
            # Optional : apply MinMaxscaling to inputs
            scaler = MinMaxScaling(ff)
            ff = scaler.fit_transform(ff, ff=False)
            target = scaler.fit_transform(target, ff=True)

            # output prediction
            pred = model(ff, skull, tinput)
            dice = batch_metrics(pred, target)

            dice_score.append(dice)
            pred_list.append(pred.squeeze().detach().cpu())
            target_list.append(target)
            
            toc = time.time()
            time_remain = datetime.timedelta(seconds=(len(test_dataloader)-i)*(toc-tic))
            
            sys.stdout.write(
            "\r[Batch %d/%d] [Dice : %f] [Time left : %f]"
            % (
                i+1,
                len(test_dataloader),
                dice,
                time_remain
                )
            )

    # result list flattening
    target_list = [item for sublist in target_list for item in sublist]
    pred_list = [item for sublist in pred_list for item in sublist]

    # display the test results
    print("================================")
    print("Mean dice score :",np.mean(dice_score))
    print("Median dice score :",np.median(dice_score))
    print("Std :",np.std(dice_score))

    return target_list, pred_list, dice_score

if __name__ == "__main__":

    PATH = "./%s/epoch_%d.pth" % (opt.run_name, opt.run_epoch)

    test_dataloader = load_dataset(opt.modality, test=True, test_batch_size=opt.batch_size)
    device = torch.device('cuda') if opt.cuda else 'cpu'

    # Define model
    model = EncoderUNet2x()
    if opt.parallel: model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(PATH, map_location=device))
    model.to(device)

    pred, target, score = eval(model, test_dataloader)
    
    # optional : save result images
    savefig(target, opt.run_name, thres=False)