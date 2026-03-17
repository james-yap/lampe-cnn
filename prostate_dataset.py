import torch 
from torch.utils.data import Dataset
from scipy.io import loadmat
import numpy as np
import random

class ProstateDataset(Dataset):
    
    def __init__(self, root_dir):
        self.samples = []
        
        classes = ["IDC", "HGC", "LGC", "Healthy"]
        
        label_map = {
            "IDC": 0,
            "HGC": 1,
            "LGC": 2,
            "Healthy": 3
        }
        
        for cls in classes:
            
            data = loadmat(f"{root_dir}/{cls}_cut_bulk_data.mat")
            
            shg = data[f"{cls}_SHG_cut"]
            bg1450 = data[f"{cls}_1450_bgsub_cut"]
            bg1668 = data[f"{cls}_1668_bgsub_cut"]
            
            n = shg.shape[2]
            
            for i in range(n):
                self.samples.append((cls, i))
                
        self.root_dir = root_dir
        self.label_map = label_map
        
        self.mean = torch.tensor([0.4997, 0.5073, 0.5064]).view(3, 1, 1)
        self.std = torch.tensor([0.0002, 0.0118, 0.0092]).view(3, 1, 1)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        
        cls, i = self.samples[idx]
        
        data = loadmat(f"{self.root_dir}/{cls}_cut_bulk_data.mat")
        
        shg = data[f"{cls}_SHG_cut"][:, :, i]
        bg1450 = data[f"{cls}_1450_bgsub_cut"][:, :, i]
        bg1668 = data[f"{cls}_1668_bgsub_cut"][:, :, i]
        
        image = np.stack([shg, bg1450, bg1668], axis=0)
        
        image = image.astype(np.float32) / 65535.0
        image = torch.tensor(image)
        
        image = (image - self.mean) / self.std
        
        # Augmentation
        if cls in ["LGC", "Healthy"]:
            k = random.randint(0, 3)
            image = np.rot90(image, k, axes=(1, 2)).copy()
            
        label = self.label_map[cls]
        
        return image, label
