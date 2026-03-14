import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SlidingWindowDataset(Dataset[torch.Tensor]):
    """
    A custom PyTorch Dataset that implements a sliding window approach to extract patches
    from a 3D image matrix (H, W, N samples).
    """
    def __init__(self, image_matrix: np.ndarray, window_size: int = 224, stride: int = 96):
        """
        Args:
            image_matrix (numpy array): The input image matrix of shape (H, W, N samples).
            window_size (int): The size of the sliding window (default: 224).
            stride (int): The stride of the sliding window (default: 96).
                          Smaller stride = more overlap = more samples.
        """
        self.image_matrix = image_matrix
        self.window_size = window_size
        self.stride = stride
        
        # Expecting shape (512, 512, N)
        self.H, self.W, self.num_samples = image_matrix.shape

        # Pre-compute all top-left (y, x) coordinates for our patches
        self.coords: list[tuple[int, int]] = []
        for y in range(0, self.H - self.window_size + 1, self.stride):
            for x in range(0, self.W - self.window_size + 1, self.stride):
                self.coords.append((y, x))
        
        self.num_patches_per_sample = len(self.coords)
    
    def __len__(self):
        return self.num_patches_per_sample * self.num_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        # 1. Figure out which sample and which coordinate this index belongs to
        sample_idx = idx // self.num_patches_per_sample
        coord_idx = idx % self.num_patches_per_sample
        
        y, x = self.coords[coord_idx]
        
        # 2. Extract the 224x224 patch from the specific sample
        patch = self.image_matrix[y:y+self.window_size, x:x+self.window_size, sample_idx]
        
        # 3. Convert to PyTorch tensor
        patch_tensor = torch.from_numpy(patch).float()
        
        # 4. Add a channel dimension so PyTorch Convs can process it
        # Shape goes from (224, 224) -> (1, 224, 224)
        patch_tensor = patch_tensor.unsqueeze(0)
        
        return patch_tensor