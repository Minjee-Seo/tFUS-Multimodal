import h5py
from torch.utils.data import Dataset, DataLoader, Subset
import torch
from utils import create_split
from natsort import natsorted

class Data(Dataset):
    def __init__(self, path=".", med="mri", eval=False):

        self.path = path
        self.split = 'valid' if eval else 'train'

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
    
def load_dataset(modality='ct', train_batch_size=8, valid_batch_size=8, test=False, test_batch_size=None):
    
    if test:
        test_set = Data(med=modality, eval=True)
        test_dataloader = DataLoader(
            test_set,
            batch_size = test_batch_size,
            shuffle=False
        )
        return test_dataloader
    
    train_set = Data(med=modality, eval=False)
    train_indices, valid_indices = create_split(8, 0.5)
    
    train_data = Subset(train_set, train_indices)
    valid_data = Subset(train_set, valid_indices)
    
    train_dataloader = DataLoader(
        train_data,
        batch_size = train_batch_size,
        shuffle = True
    )

    valid_dataloader = DataLoader(
        valid_data,
        batch_size = valid_batch_size,
        shuffle = False
    )
    
    return train_dataloader, valid_dataloader