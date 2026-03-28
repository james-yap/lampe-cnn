"""
Class-Balanced Architecture: Sliding Window with ResNet18 Backbone,
per-channel z-score normalization, geometric augmentation, and
weighted random oversampling to address class imbalance.

Based directly on standardized.py with the following additions:
- On-the-fly class imbalance detection and reporting
- Per-patch sample_weights tensor for WeightedRandomSampler
- Geometric augmentation (hflip, vflip, rot90) on training split only
- Optional layer3 unfreezing in the model factory
"""

from collections import Counter

import torch
from torch.utils.data import Dataset
from torch import nn
from torchvision import models

from shared.mat_reader import MatReader
from shared.constants import CLASS_NAMES

Datapoint = tuple[torch.Tensor, int, str]  # (patch tensor, class label, patient id)


class ClassBalancedDataset(Dataset[Datapoint]):
    """
    Sliding window dataset with per-channel z-score normalization,
    geometric augmentation (train split only), and per-patch sample_weights
    for use with WeightedRandomSampler to address class imbalance.
    """

    mat_reader: MatReader
    window_size: int
    stride: int
    num_patches_per_fov: int
    top_left_coords: list[tuple[int, int]]
    mean: torch.Tensor
    std: torch.Tensor
    train: bool
    sample_weights: torch.Tensor  # one weight per patch, for WeightedRandomSampler

    def __init__(
        self,
        mat_reader: MatReader,
        eff_fov_indices: list[int],
        factor: int = 5,
        train: bool = True,
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
            train (bool): If True, enables geometric augmentation. Set to False for val/test.
            mean_override (torch.Tensor | None): Optional pre-computed mean for normalization.
                                                 Use on val/test sets to prevent data leakage.
            std_override (torch.Tensor | None): Optional pre-computed std for normalization.
                                                Use on val/test sets to prevent data leakage.
        """
        self.mat_reader = mat_reader
        self.eff_fov_indices = eff_fov_indices
        self.train = train

        self.window_size = 224
        height, width = mat_reader.get_height_width()
        self.stride = (height - self.window_size) // (factor - 1)

        # Pre-compute all top-left (y, x) coordinates for our patches
        self.top_left_coords: list[tuple[int, int]] = []
        for y in range(0, height - self.window_size + 1, self.stride):
            for x in range(0, width - self.window_size + 1, self.stride):
                self.top_left_coords.append((y, x))

        self.num_patches_per_fov = len(self.top_left_coords)

        # --- On-the-fly class imbalance detection ---
        class_counts: Counter[int] = Counter(
            int(mat_reader.class_labels[i]) for i in eff_fov_indices
        )
        total_fovs = len(eff_fov_indices)
        split_label = "train" if train else "val"
        print(f"  [{split_label}] Class distribution ({total_fovs} FOVs):")
        for cls_idx in range(len(CLASS_NAMES)):
            count = class_counts.get(cls_idx, 0)
            pct = 100.0 * count / total_fovs if total_fovs > 0 else 0.0
            print(f"    {CLASS_NAMES[cls_idx]:8s}: {count:4d} FOVs ({pct:.1f}%)")

        # --- Per-patch sample weights (inverse FOV-level class frequency) ---
        # Every patch from a FOV of class c gets weight = 1 / num_fovs_in_class_c.
        # Weighting at the FOV level (not the patch level) avoids inflating the
        # effective sample size: each FOV's 25 patches share one class weight.
        patch_weights: list[float] = []
        for fov_idx in eff_fov_indices:
            cls = int(mat_reader.class_labels[fov_idx])
            w = 1.0 / class_counts[cls] if class_counts[cls] > 0 else 1.0
            patch_weights.extend([w] * self.num_patches_per_fov)
        self.sample_weights = torch.tensor(patch_weights, dtype=torch.float)

        # --- Per-channel z-score normalization ---
        if mean_override is not None and std_override is not None:
            self.mean = mean_override
            self.std = std_override
        else:
            num_channels = mat_reader.get_num_channels()
            channel_sums = torch.zeros(num_channels)
            channel_squared_sums = torch.zeros(num_channels)
            num_pixels = 0
            for fov_idx in eff_fov_indices:
                image = torch.from_numpy(
                    mat_reader.images[fov_idx]
                ).float()  # (n modalities, H, W)
                channel_sums += image.sum(dim=[1, 2])
                channel_squared_sums += (image**2).sum(dim=[1, 2])
                num_pixels += image.size(1) * image.size(2)
            self.mean = channel_sums / num_pixels
            variance = torch.clamp(
                channel_squared_sums / num_pixels - self.mean**2, min=0.0
            )
            self.std = torch.sqrt(variance)
            # Replace near-zero std (constant channels) with 1 to prevent division by zero
            self.std[self.std < 1e-8] = 1.0

    def __len__(self) -> int:
        return len(self.eff_fov_indices) * self.num_patches_per_fov

    def __getitem__(self, idx: int) -> Datapoint:
        fov_idx = self.eff_fov_indices[idx // self.num_patches_per_fov]
        patch_idx = idx % self.num_patches_per_fov
        y, x = self.top_left_coords[patch_idx]

        patch = self.mat_reader.images[
            fov_idx, :, y : y + self.window_size, x : x + self.window_size
        ]
        patch_tensor = torch.from_numpy(patch).float()  # (C, 224, 224)
        class_label = int(self.mat_reader.class_labels[fov_idx])
        patient_id = str(self.mat_reader.patient_ids[fov_idx])

        # Standardize using pre-computed train-split statistics
        patch_tensor = (patch_tensor - self.mean[:, None, None]) / self.std[
            :, None, None
        ]

        # Geometric augmentation — train split only.
        # Horizontal flip, vertical flip, and 90°/180°/270° rotation are valid for
        # SRS microscopy because tissue sections have no canonical orientation and
        # these transforms preserve all quantitative Raman spectral channel information.
        if self.train:
            if torch.rand(1).item() > 0.5:
                patch_tensor = torch.flip(patch_tensor, dims=[2])  # horizontal flip
            if torch.rand(1).item() > 0.5:
                patch_tensor = torch.flip(patch_tensor, dims=[1])  # vertical flip
            k = int(torch.randint(0, 4, (1,)).item())
            if k > 0:
                patch_tensor = torch.rot90(patch_tensor, k=k, dims=[1, 2])

        return patch_tensor, class_label, patient_id


def get_model(num_classes: int = 4, unfreeze_layer3: bool = False) -> nn.Module:
    """
    ResNet18 with pretrained ImageNet weights, partially fine-tuned.

    Args:
        num_classes (int): Number of output classes (default: 4).
        unfreeze_layer3 (bool): If True, also unfreeze layer3 in addition to layer4.
                                Keep False for the first run to isolate the effect of
                                class balancing; set True for follow-up experiments
                                where increased effective dataset diversity from
                                augmentation and oversampling may justify a larger
                                fine-tuning surface.

    Architecture Justifications:
    1. ResNet18: Deep enough to learn complex spatial hierarchies, but small enough
       to prevent massive overfitting on sliding window crops (can be highly correlated).
    2. Pretrained Weights: Faster convergence, better low-level edge/texture detectors.
    3. Custom Head with Dropout(0.5): Regularizes the highly correlated patch representations.
    """
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Freeze entire network
    for param in model.parameters():
        param.requires_grad = False

    # Always unfreeze final residual block (layer4)
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Optionally unfreeze layer3 for more fine-tuning capacity
    if unfreeze_layer3:
        for param in model.layer3.parameters():
            param.requires_grad = True

    # Replace classification head
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(  # type: ignore
        nn.Dropout(0.5), nn.Linear(num_ftrs, num_classes)
    )

    return model
