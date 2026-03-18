import os, yaml
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def _load_config():
    if os.path.exists("config/config.yaml"):
        with open("config/config.yaml", "r") as f:
            return yaml.safe_load(f)
    else:
        raise FileNotFoundError("ERROR: config not found at config/config.yaml .\nEnsure you running from project's root directory")

def get_data_splits():
    config = _load_config()
    data = pd.read_csv(config["paths"]["data"])
    
    train_df = data[data['Usage'] == 'Training'].reset_index(drop=True)
    val_df   = data[data['Usage'] == 'PublicTest'].reset_index(drop=True)
    test_df  = data[data['Usage'] == 'PrivateTest'].reset_index(drop=True)
    
    return train_df, val_df, test_df

class FERDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.itoc = {0:'Anger', 
                     1:'Disgust', 
                     2:'Fear', 
                     3:'Happy', 
                     4:'Sad', 
                     5:'Surprise', 
                     6:'Neutral'}
        self.ctoi = {v:k for k, v in self.itoc.items()}
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pixels = self.df.loc[idx, 'pixels']
        img = np.fromstring(pixels, dtype=np.uint8, sep=' ').reshape(48, 48)
        img = np.stack((img,) * 3, axis=-1)   # (48, 48, 3)

        label = self.df.loc[idx, 'emotion']

        if self.transform:
            img = self.transform(img)
        return img, label

    @classmethod
    def dataloader(cls, batch_size:int, shuffle:bool):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        return DataLoader(cls, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
    