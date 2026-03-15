import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from shared.mat_reader import MatReader

Datapoint = tuple[torch.Tensor, int] # (patch tensor, patient index)

class SlidingWindowDataset(Dataset[Datapoint]):
    """
    A custom PyTorch Dataset that implements a sliding window approach to extract patches
    from a 3D image matrix (H, W, N samples).
    """
    
    mat_reader: MatReader
    window_size: int
    stride: int
    num_fovs: int
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
        
        data = mat_reader.get_data()
        example_datapoint = data[0] # Assuming all samples have the same height and width
        height, width = example_datapoint.shape[1:3] 
        # Pre-compute all top-left (y, x) coordinates for our patches
        self.top_left_coords: list[tuple[int, int]] = []
        for y in range(0, height - self.window_size + 1, self.stride):
            for x in range(0, width - self.window_size + 1, self.stride):
                self.top_left_coords.append((y, x))
        
        self.num_patches_per_fov = len(self.top_left_coords)
        
        self.num_fovs = 0
        for patient in data:
            self.num_fovs += patient.shape[0] # number of samples for this patient
    
    def __len__(self):
        return self.num_patches_per_fov * self.num_fovs
    
    def __getitem__(self, idx: int) -> Datapoint:
        seen_datapoints = 0
        for patient_idx, patient in enumerate(self.mat_reader.get_data()):
            num_fovs = patient.shape[0]
            num_datapoints = num_fovs * self.num_patches_per_fov
            
            seen_datapoints += num_datapoints
            
            if idx < seen_datapoints:
                fov_idx = idx // self.num_patches_per_fov
                patch_idx = idx % self.num_patches_per_fov
                
                y, x = self.top_left_coords[patch_idx]
                patch = patient[fov_idx, y:y+self.window_size, x:x+self.window_size, :]
                patch_tensor = torch.from_numpy(patch).float().permute(2, 0, 1) # (n modalities, 224, 224)
                
                # TODO: missing classification label
                return patch_tensor, patient_idx
        
            idx -= seen_datapoints
            seen_datapoints = 0
        
        raise IndexError(f"Index {idx} out of range for dataset of length {self.__len__()}")