from torch.utils.data import DataLoader, Subset
from dataset import Data
from utils import create_split

def load_dataset(modality='ct', train_batch_size=8, valid_batch_size=8, test=False, test_batch_size=None):
    
    if test:
        test_set = Data(modality, eval=True)
        test_dataloader = DataLoader(
            test_set,
            batch_size = test_batch_size,
            shuffle=False
        )
        return test_dataloader
    
    train_set = Data(modality, eval=False)
    train_indices, valid_indices = create_split(8, 0.2)
    
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