from torch.utils.data import Dataset
import torch
import h5py
import numpy as np
import os
from glob import glob
from natsort import natsorted

class Data(Dataset):
    def __init__(self, path="./tFUS_MM_data", med="mri", eval=False):

        self.path = path
        self.split = 'test' if eval else 'train'

        self.x = self._load_data("ff", self.split)
        self.s = self._load_data(med, self.split)
        self.y = self._load_data("sk", self.split)
        self.t = self._load_data("tin", self.split)
        
        self.index = natsorted(list(self.x.keys()))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ix = self.index[idx]
        X = torch.Tensor(self.x[ix][()]).float()
        Y = torch.Tensor(self.y[ix][()]).float()
        S = torch.Tensor(self.s[ix][()]).float()
        T = torch.Tensor(self.t[ix][()]).float()
        
        return {"A":X, "B":Y, "S":S, "T":T}
    
    def _load_data(self, name, split):

        path = self.path + '/%s_%s.hdf5'%(name,split)
        data = h5py.File(path,'r')

        return data