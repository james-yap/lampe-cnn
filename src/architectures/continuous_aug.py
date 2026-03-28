"""
Continuous Augmentation Architecture: Ordinal K-1 encoding with infinite
augmentation space via continuous rotation, random crop translation, and
additive Gaussian noise.

Builds on regularized.py (same ordinal encoding, phased unfreezing, weight
decay, ReduceLROnPlateau) and replaces the fixed-stride discrete augmentation
with three operations that together make the probability of seeing an exact
duplicate training tensor effectively zero:

1. Random crop (continuous translation):
   Instead of a pre-computed factor×factor stride grid, each training call
   samples a uniformly random top-left corner from the full valid crop range.
   The validation split still uses the fixed grid for reproducibility.

2. Continuous rotation via TF.rotate (U(0°, 360°)):
   Replaces rot90 (4 discrete values) with a uniform random angle in [0, 360).
   Bilinear interpolation is used; corners are padded with 0.0 (the
   mean-normalised background value after z-score normalisation).

3. Additive Gaussian noise (σ=0.02, post-normalisation, train only):
   A small independent noise sample is added to every training tensor. After
   z-score normalisation, typical inter-class signal differences span several
   standardised units; σ=0.02 is approximately 1–2% of that range. This
   guarantees every tensor is unique across epochs while preserving
   discriminatory spectral texture. The parameter is conservative and should
   be reduced to σ=0.01 or removed entirely if val loss degrades vs. the
   regularized baseline.

Pigeonhole fix:
   The current discrete pipeline (hflip × vflip × rot90) yields at most
   2 × 2 × 4 = 16 unique augmented variants per patch. With WeightedRandom-
   Sampler oversampling the Healthy class by ~9.5×, the same patch can appear
   dozens of times per epoch in identical form — a direct memorisation path.
   This architecture closes that gap by making the augmentation space infinite.

get_model() is identical to regularized.get_model() — re-exported unchanged.
"""

from collections import Counter

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torch import nn  # noqa: F401 — re-exported via regularized

from shared.mat_reader import MatReader
from shared.constants import CLASS_NAMES

# Re-export all public symbols from regularized so the CLI can use a single
# import surface regardless of which architecture it chooses.
from architectures.regularized import (  # noqa: F401
    OrdinalDatapoint,
    encode_ordinal,
    decode_ordinal,
    get_model,
)


class ContinuousAugDataset(Dataset[OrdinalDatapoint]):
    """
    Sliding window dataset with per-channel z-score normalisation,
    continuous geometric augmentation (train split only), per-patch
    sample_weights for WeightedRandomSampler, and ordinal K-1 label encoding.

    Differences from OrdinalDataset:
    - Train translation: random crop origin sampled uniformly each call
      (instead of a pre-computed fixed stride grid).
    - Train rotation: continuous U(0°, 360°) with bilinear interpolation
      (instead of discrete rot90 with 4 choices).
    - Train noise: additive Gaussian σ=0.02 applied after normalisation
      (guarantees every tensor is unique).
    - Validation: identical to OrdinalDataset — fixed stride grid for
      reproducible evaluation.

    Returns 4-tuples: (patch, ordinal_label_vector, class_label_int, patient_id).
    """

    mat_reader: MatReader
    window_size: int
    stride: int
    num_patches_per_fov: int
    top_left_coords: list[tuple[int, int]]
    _max_y: int
    _max_x: int
    mean: torch.Tensor
    std: torch.Tensor
    train: bool
    sample_weights: torch.Tensor

    def __init__(
        self,
        mat_reader: MatReader,
        eff_fov_indices: list[int],
        factor: int = 5,
        train: bool = True,
        mean_override: torch.Tensor | None = None,
        std_override: torch.Tensor | None = None,
    ) -> None:
        """
        Args:
            mat_reader:      Loaded MatReader instance.
            eff_fov_indices: FOV indices for this split (train or val).
            factor:          Patches per dimension (e.g. 5 → 5×5 = 25 patches).
                             Controls the validation grid density and the
                             effective dataset length; window size is fixed at 224.
            train:           If True, applies continuous augmentation. False for val.
            mean_override:   Pre-computed train-split mean (use on val to prevent leakage).
            std_override:    Pre-computed train-split std (use on val to prevent leakage).
        """
        self.mat_reader = mat_reader
        self.eff_fov_indices = eff_fov_indices
        self.train = train

        self.window_size = 224
        height, width = mat_reader.get_height_width()
        self.stride = (height - self.window_size) // (factor - 1)

        # Fixed grid — used for validation reproducibility and dataset length sizing
        self.top_left_coords: list[tuple[int, int]] = []
        for y in range(0, height - self.window_size + 1, self.stride):
            for x in range(0, width - self.window_size + 1, self.stride):
                self.top_left_coords.append((y, x))
        self.num_patches_per_fov = len(self.top_left_coords)

        # Inclusive upper bounds for random crop sampling (train split)
        self._max_y = height - self.window_size
        self._max_x = width - self.window_size

        # --- Class distribution report ---
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
        patch_weights: list[float] = []
        for fov_idx in eff_fov_indices:
            cls = int(mat_reader.class_labels[fov_idx])
            w = 1.0 / class_counts[cls] if class_counts[cls] > 0 else 1.0
            patch_weights.extend([w] * self.num_patches_per_fov)
        self.sample_weights = torch.tensor(patch_weights, dtype=torch.float)

        # --- Per-channel z-score normalisation ---
        if mean_override is not None and std_override is not None:
            self.mean = mean_override
            self.std = std_override
        else:
            num_channels = mat_reader.get_num_channels()
            channel_sums = torch.zeros(num_channels)
            channel_squared_sums = torch.zeros(num_channels)
            num_pixels = 0
            for fov_idx in eff_fov_indices:
                image = torch.from_numpy(mat_reader.images[fov_idx]).float()
                channel_sums += image.sum(dim=[1, 2])
                channel_squared_sums += (image**2).sum(dim=[1, 2])
                num_pixels += image.size(1) * image.size(2)
            self.mean = channel_sums / num_pixels
            variance = torch.clamp(
                channel_squared_sums / num_pixels - self.mean**2, min=0.0
            )
            self.std = torch.sqrt(variance)
            self.std[self.std < 1e-8] = 1.0

    def __len__(self) -> int:
        return len(self.eff_fov_indices) * self.num_patches_per_fov

    def __getitem__(self, idx: int) -> OrdinalDatapoint:
        fov_idx = self.eff_fov_indices[idx // self.num_patches_per_fov]
        class_label = int(self.mat_reader.class_labels[fov_idx])
        patient_id = str(self.mat_reader.patient_ids[fov_idx])

        if self.train:
            # Continuous translation: sample a random top-left corner each call
            y = int(torch.randint(0, self._max_y + 1, (1,)).item())
            x = int(torch.randint(0, self._max_x + 1, (1,)).item())
        else:
            # Fixed grid for reproducible validation
            patch_idx = idx % self.num_patches_per_fov
            y, x = self.top_left_coords[patch_idx]

        patch = self.mat_reader.images[
            fov_idx, :, y : y + self.window_size, x : x + self.window_size
        ]
        patch_tensor = torch.from_numpy(patch).float()  # (C, 224, 224)

        # Z-score normalisation using train-split statistics
        patch_tensor = (patch_tensor - self.mean[:, None, None]) / self.std[
            :, None, None
        ]

        if self.train:
            # Flips (discrete — still valid and cheap)
            if torch.rand(1).item() > 0.5:
                patch_tensor = torch.flip(patch_tensor, dims=[2])  # hflip
            if torch.rand(1).item() > 0.5:
                patch_tensor = torch.flip(patch_tensor, dims=[1])  # vflip

            # Continuous rotation: uniform angle in [0°, 360°)
            # fill=0.0 pads corners with the mean-normalised background value
            angle = torch.rand(1).item() * 360.0
            patch_tensor = TF.rotate(
                patch_tensor,
                angle=angle,
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=[0.0],
            )

            # Additive Gaussian noise: σ=0.02 on normalised data
            # ~1–2% of typical inter-class z-score signal range; guarantees
            # every training tensor is unique, preventing exact memorisation.
            patch_tensor = patch_tensor + torch.randn_like(patch_tensor) * 0.02

        ordinal_label = encode_ordinal(class_label)
        return patch_tensor, ordinal_label, class_label, patient_id
