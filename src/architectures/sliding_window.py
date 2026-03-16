import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from shared.mat_reader import MatReader

Datapoint = tuple[torch.Tensor, int, int] # (subimage data, class label, patient id)

class SlidingWindowDataset(Dataset[Datapoint]):
    """
    A custom PyTorch Dataset that implements a sliding window approach to extract patches
    from a 3D image matrix (H, W, N samples).
    """
    
    mat_reader: MatReader
    window_size: int
    stride: int
    num_patches_per_fov: int
    top_left_coords: list[tuple[int, int]] # all top-left (y, x) coordinates for our patches

    def __init__(self, mat_reader: MatReader, window_size: int = 224, stride: int = 96):
        """
        Args:
            mat_reader (MatReader): An instance of MatReader to load the image matrix.
            window_size (int): The size of the sliding window (default: 224).
            stride (int): The stride of the sliding window (default: 96).
                          Smaller stride = more overlap = more samples.
        """
        self.mat_reader = mat_reader
        self.window_size = window_size
        self.stride = stride
        
        height, width = mat_reader.get_dims()[2], mat_reader.get_dims()[3]

        # Pre-compute all top-left (y, x) coordinates for our patches
        self.top_left_coords: list[tuple[int, int]] = []
        for y in range(0, height - self.window_size + 1, self.stride):
            for x in range(0, width - self.window_size + 1, self.stride):
                self.top_left_coords.append((y, x))
        
        self.num_patches_per_fov = len(self.top_left_coords)
    
    def __len__(self):
        return self.mat_reader.get_dims()[0] * self.num_patches_per_fov
    
    def __getitem__(self, idx: int) -> Datapoint:
        fov_idx = idx // self.num_patches_per_fov
        patch_idx = idx % self.num_patches_per_fov
        y, x = self.top_left_coords[patch_idx]

        patch = self.mat_reader.images[fov_idx, :, y:y+self.window_size, x:x+self.window_size]
        patch_tensor = torch.from_numpy(patch).float() # (n modalities, 224, 224)
        class_label = int(self.mat_reader.class_labels[fov_idx])
        patient_idx = int(self.mat_reader.patient_ids[fov_idx])
        return patch_tensor, class_label, patient_idx

class EarlyStopping:
    """
    Stateful class to track validation loss and determine when to stop training early.
    """
    def __init__(self, patience=5, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True