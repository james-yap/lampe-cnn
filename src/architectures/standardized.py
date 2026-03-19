"""
Geometric Augmentation Architecture: Sliding Window with ResNet18 Backbone
"""

import torch
from torch.utils.data import Dataset
from torch import nn
from torchvision import models

from shared.mat_reader import MatReader

Datapoint = tuple[torch.Tensor, int, str]  # (subimage data, class label, patient id)


class StandardizedDataset(Dataset[Datapoint]):
    """
    A custom PyTorch Dataset that implements a sliding window approach to extract patches
    from a 3D image matrix (H, W, N samples).
    """

    mat_reader: MatReader
    window_size: int
    stride: int
    num_patches_per_fov: int
    top_left_coords: list[
        tuple[int, int]
    ]  # all top-left (y, x) coordinates for our patches
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(
        self,
        mat_reader: MatReader,
        eff_fov_indices: list[int],
        factor: int = 5,
        mean_override: torch.Tensor | None = None,
        std_override: torch.Tensor | None = None,
    ):
        """
        Args:
            mat_reader (MatReader): An instance of MatReader to load the image matrix.
            eff_fov_indices (list[int]): List of FOV indices to include in the dataset.
                                     The rest are ignored, allowing us to create
                                     train/val/test splits at the FOV level.
            factor (int): The number of patches per dimension
                          (e.g., factor=5 means 5x5=25 patches per FOV).
                          Window size is fixed at 224 (optimal for ResNet),
                          so stride is computed as (image_dim - window_size) // (factor - 1).
            mean_override (torch.Tensor | None): Optional pre-computed mean for normalization.
                                                 Use on val/test sets to prevent data leakage.
            std_override (torch.Tensor | None): Optional pre-computed std for normalization.
        """
        self.mat_reader = mat_reader
        self.eff_fov_indices = eff_fov_indices

        self.window_size = 224
        height, width = mat_reader.get_height_width()
        self.stride = (height - self.window_size) // (
            factor - 1
        )  # (height - window_size) // (factor - 1)

        # Pre-compute all top-left (y, x) coordinates for our patches
        self.top_left_coords: list[tuple[int, int]] = []
        for y in range(0, height - self.window_size + 1, self.stride):
            for x in range(0, width - self.window_size + 1, self.stride):
                self.top_left_coords.append((y, x))

        self.num_patches_per_fov = len(self.top_left_coords)

        # Computing mean and std for normalization (standardization)
        if mean_override is not None and std_override is not None:
            self.mean = mean_override
            self.std = std_override
        else:
            num_channels = mat_reader.get_num_channels()
            channel_sums = torch.zeros(num_channels)  # sum for each modality
            channel_squared_sums = torch.zeros(
                num_channels
            )  # sum of squares for each modality
            num_pixels = 0
            for fov_idx in eff_fov_indices:
                image = torch.from_numpy(
                    mat_reader.images[fov_idx]
                ).float()  # (n modalities, H, W)
                channel_sums += image.sum(
                    dim=[1, 2]
                )  # sum over H and W for each modality
                channel_squared_sums += (image**2).sum(dim=[1, 2])  # sum of squares
                num_pixels += image.size(1) * image.size(2)  # total pixels per modality
            self.mean = channel_sums / num_pixels  # (n modalities,)
            variance = torch.clamp(
                channel_squared_sums / num_pixels - self.mean**2, min=0.0
            )
            self.std = torch.sqrt(variance)  # (n modalities,)
            # Replace near-zero std (constant channels) with 1 to prevent division by zero
            self.std[self.std < 1e-8] = 1.0

    def __len__(self):
        return (
            len(self.eff_fov_indices) * self.num_patches_per_fov
        )  # total number of patches across all FOVs

    def __getitem__(self, idx: int) -> Datapoint:
        fov_idx = self.eff_fov_indices[idx // self.num_patches_per_fov]
        patch_idx = idx % self.num_patches_per_fov
        y, x = self.top_left_coords[patch_idx]

        patch = self.mat_reader.images[
            fov_idx, :, y : y + self.window_size, x : x + self.window_size
        ]
        patch_tensor = torch.from_numpy(patch).float()  # (n modalities, 224, 224)
        class_label = int(self.mat_reader.class_labels[fov_idx])
        patient_id = self.mat_reader.patient_ids[fov_idx]

        # Standardize the patch using the pre-computed mean and std
        patch_tensor = (patch_tensor - self.mean[:, None, None]) / self.std[
            :, None, None
        ]

        return patch_tensor, class_label, patient_id


def get_model(num_classes=4) -> nn.Module:
    """
    Architecture Justifications:
    1. ResNet18: Deep enough to learn complex spatial hierarchies, but small enough
       to prevent massive overfitting on sliding window crops (can be highly correlated).
    2. Pretrained Weights (Transfer Learning): Initializes with ImageNet,
       converging faster and requiring less data to learn fundemental edge/texture detectors.
    3. Custom Head with Dropout: Replace the final Fully Connected Layer. Strong dropout
       to force model to rely on distributed representations (due to overlapping nature
       of sliding windows).
    """

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # freeze entire network
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze final block (layer4)
    for param in model.layer4.parameters():
        param.requires_grad = True
    # note: In ResNet18, layer4 is a BasicBlock with two convolutional layers.
    # note: for more info, read up on `BasicBlock` and `Bottleneck`

    # replace classification head
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(  # type: ignore
        nn.Dropout(0.5), nn.Linear(num_ftrs, num_classes)
    )

    return model
