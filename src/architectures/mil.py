"""
Multiple Instance Learning (MIL) Architecture: attention-based FOV-level
classification using ResNet18 feature extraction and softmax attention pooling.

Addresses the fundamental weak-supervision problem in patch-based approaches:
every patch in a Healthy FOV is labeled Healthy, but some 224×224 windows may
contain tissue features that resemble early-grade cancer. Training a patch
classifier with these inherited labels teaches the model to hedge toward the
majority class (HGC collapse observed in regularized and continuous_aug runs).

MIL treats each FOV as a bag and each patch as an instance. The bag label
(ground truth from pathologist annotation) is used for training; instance
labels are never seen by the loss function.

Architecture (Ilse et al. 2018, "Attention-based Deep Multiple Instance Learning"):

    Stage 1 — Shared backbone (ResNet18, ImageNet pre-trained):
        Each of the N patches in the bag is processed independently.
        Output: (N, 512) feature vectors from the global average pooling layer.

    Stage 2 — Attention aggregation:
        a_k  = softmax_k( W_att · tanh(V_att · z_k) )   scalar for patch k
        z_bag = Σ_k  a_k * z_k                           weighted bag descriptor (512,)

    Stage 3 — Classification head:
        Dropout(0.5) + Linear(512, num_classes) → CrossEntropyLoss (4 classes).

Why CrossEntropyLoss (not BCEWithLogitsLoss):
    The ordinal K-1 encoding is dropped. Each FOV is classified directly with
    a single argmax over 4 class logits, removing the conjunction bias that
    caused ordinal hedging in regularized and continuous_aug architectures.

Dataset:
    MILDataset returns one FOV at a time (one bag of patches), not one patch.
    __len__ returns the number of FOVs, so WeightedRandomSampler weights and
    DataLoader iteration are at the FOV granularity.

Per-patch augmentation mirrors ContinuousAugDataset: random crop, continuous
rotation U(0°, 360°), Gaussian noise σ=0.02. Applied independently to each
patch within the bag (train split only). Val uses the fixed stride grid.

Phased unfreezing:
    When freeze_all=True the entire backbone (including layer4) starts frozen.
    OptimizerEngine.maybe_transition_phase() detects "layer4" in named_parameters
    and unfreezes backbone.layer4 at epoch head_only_epochs. The attention
    modules and classifier are trainable from epoch 0.
"""

from collections import Counter

import torch
import torchvision.transforms.functional as TF
from torch import nn
from torch.utils.data import Dataset
from torchvision import models

from shared.mat_reader import MatReader
from shared.constants import CLASS_NAMES

# One bag per FOV: (N_patches, C, H, W) tensor, integer class label, patient ID
MILDatapoint = tuple[torch.Tensor, int, str]


class MILDataset(Dataset[MILDatapoint]):
    """
    FOV-level dataset for multiple instance learning.

    Each item is a full FOV represented as a bag of N_patches 224×224 patches.
    Unlike patch-level datasets, __len__ returns the number of FOVs so that
    WeightedRandomSampler operates at the FOV granularity (one weight per FOV).

    Train split: random-crop + continuous-rotation + Gaussian noise applied
    independently to each patch in the bag.
    Val split: fixed stride grid, no augmentation (same as ContinuousAugDataset).
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
    sample_weights: torch.Tensor  # one weight per FOV

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
            factor:          Patches per spatial dimension (factor² patches per FOV).
                             Controls patch grid density; window_size fixed at 224.
            train:           If True, apply per-patch continuous augmentation.
            mean_override:   Pre-computed train-split channel mean (use on val).
            std_override:    Pre-computed train-split channel std (use on val).
        """
        self.mat_reader = mat_reader
        self.eff_fov_indices = eff_fov_indices
        self.train = train

        self.window_size = 224
        height, width = mat_reader.get_height_width()
        self.stride = (height - self.window_size) // (factor - 1)

        # Fixed grid — used for validation reproducibility
        self.top_left_coords: list[tuple[int, int]] = []
        for y in range(0, height - self.window_size + 1, self.stride):
            for x in range(0, width - self.window_size + 1, self.stride):
                self.top_left_coords.append((y, x))
        self.num_patches_per_fov = len(self.top_left_coords)

        # Inclusive upper bounds for random crop (train split only)
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

        # --- Per-FOV sample weights (inverse class frequency, one per FOV) ---
        fov_weights: list[float] = []
        for fov_idx in eff_fov_indices:
            cls = int(mat_reader.class_labels[fov_idx])
            w = 1.0 / class_counts[cls] if class_counts[cls] > 0 else 1.0
            fov_weights.append(w)
        self.sample_weights = torch.tensor(fov_weights, dtype=torch.float)

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
        """Returns number of FOVs (not patches) — WeightedRandomSampler operates per FOV."""
        return len(self.eff_fov_indices)

    def __getitem__(self, idx: int) -> MILDatapoint:
        fov_idx = self.eff_fov_indices[idx]
        class_label = int(self.mat_reader.class_labels[fov_idx])
        patient_id = str(self.mat_reader.patient_ids[fov_idx])

        patch_tensors: list[torch.Tensor] = []
        for patch_idx in range(self.num_patches_per_fov):
            if self.train:
                # Continuous translation: random crop origin each call
                y = int(torch.randint(0, self._max_y + 1, (1,)).item())
                x = int(torch.randint(0, self._max_x + 1, (1,)).item())
            else:
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
                if torch.rand(1).item() > 0.5:
                    patch_tensor = torch.flip(patch_tensor, dims=[2])  # hflip
                if torch.rand(1).item() > 0.5:
                    patch_tensor = torch.flip(patch_tensor, dims=[1])  # vflip

                angle = torch.rand(1).item() * 360.0
                patch_tensor = TF.rotate(
                    patch_tensor,
                    angle=angle,
                    interpolation=TF.InterpolationMode.BILINEAR,
                    fill=[0.0],
                )

                # Gaussian noise: σ=0.02, same as ContinuousAugDataset
                patch_tensor = patch_tensor + torch.randn_like(patch_tensor) * 0.02

            patch_tensors.append(patch_tensor)

        bag = torch.stack(patch_tensors)  # (N_patches, C, H, W)
        return bag, class_label, patient_id


class AttentionMIL(nn.Module):
    """
    Attention-based MIL model for FOV-level classification.

    Processes a batch of bags (one bag = one FOV) through a shared ResNet18
    backbone, then aggregates per-patch features via learned attention weights
    into a single FOV-level descriptor passed to the classification head.

    forward() input:  bags of shape (B, N_patches, C, H, W)
    forward() output: (B, num_classes) logits, (B, N_patches) attention weights

    The attention weights are differentiable during training (gradients flow
    through them back into the backbone in phase 2). At inference time they
    map directly to the per-patch spatial heatmap in the 5-panel visualisation.
    """

    backbone: models.ResNet
    attention_V: nn.Sequential
    attention_W: nn.Linear
    classifier: nn.Sequential

    def __init__(self, num_classes: int = 4, freeze_all: bool = False) -> None:
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT
        backbone = models.resnet18(weights=weights)

        # Replace classification head with Identity to expose 512-dim avgpool
        # features. type: ignore because fc is typed as Linear in torchvision stubs.
        backbone.fc = nn.Identity()  # type: ignore[assignment]

        # Freeze entire backbone first
        for param in backbone.parameters():
            param.requires_grad = False

        # Unfreeze layer4 unless phased unfreezing defers it to phase 2
        if not freeze_all:
            for param in backbone.layer4.parameters():
                param.requires_grad = True

        self.backbone = backbone

        # Gated attention network (Ilse et al. 2018)
        self.attention_V = nn.Sequential(nn.Linear(512, 128), nn.Tanh())
        self.attention_W = nn.Linear(128, 1)

        # Classification head on the aggregated bag descriptor
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, num_classes))

    def forward(self, bags: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            bags: (B, N_patches, C, H, W) — batch of B FOVs.

        Returns:
            logits:   (B, num_classes)
            attn_out: (B, N_patches) softmax-normalised attention weights
        """
        B, N, C, H, W = bags.shape

        # Run shared backbone on all patches simultaneously
        patches = bags.view(B * N, C, H, W)
        feats: torch.Tensor = self.backbone(patches)  # (B*N, 512)
        feats = feats.view(B, N, -1)  # (B, N, 512)

        # Attention weights
        a_V = self.attention_V(feats)  # (B, N, 128)
        a_W: torch.Tensor = self.attention_W(a_V)  # (B, N, 1)
        attn = torch.softmax(a_W, dim=1)  # (B, N, 1)

        # Attention-weighted bag descriptor
        z_bag = (attn * feats).sum(dim=1)  # (B, 512)
        logits: torch.Tensor = self.classifier(z_bag)  # (B, num_classes)
        attn_out = attn.squeeze(2)  # (B, N)

        return logits, attn_out


def get_model(num_classes: int = 4, freeze_all: bool = False) -> nn.Module:
    """
    Instantiate an AttentionMIL model.

    Args:
        num_classes: Number of output classes (default 4: Healthy, LGC, HGC, IDC).
        freeze_all:  If True, freeze the entire backbone including layer4.
                     Use with head_only_epochs > 0 for phased unfreezing via
                     OptimizerEngine.maybe_transition_phase().

    Returns:
        nn.Module — the AttentionMIL model ready for training.
    """
    return AttentionMIL(num_classes=num_classes, freeze_all=freeze_all)
