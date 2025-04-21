import ast
import argparse

def arg_list(s):
    v = ast.literal_eval(s)
    return v

def load_train_config():

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
    parser.add_argument('--wandb_pj',      type=str,   default=None,    help='Specify project name if logging with wandb')
    
    return parser.parse_args()

def load_test_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=False, help="Use GPU")
    parser.add_argument('--batch_size', type=int, default=8, help="Test batch size")
    parser.add_argument('--modality', type=str, default='mri', help="Select med.image modality (CT/MR)")
    parser.add_argument('--model', type=str, default='unet', help="Choose model to train : [ae/unet/swin]")
    parser.add_argument('--run_name', type=str, default='test_run', help="project name to load")
    parser.add_argument('--run_epoch', type=int, default=200, help="Trained epoch")
    parser.add_argument('--plot', action='store_true', default=False, help="Save result images")
    
    return parser.parse_args()