"""
Linear SVM as classification head.
Relies on pre-trained feature extractor for non-linearity.
Multi-Class Hinge Loss and L2 Regularization.

Fully frozen ResNet18 backbone as feature extractor, no fine-tuning.
Un-augmented, bad SHG removed.
Works on sparse, imbalanced data as in the original paper (Gagnon et al., 2025).
"""

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import models

from shared.mat_reader import MatReader
from shared.utils import report_class_distribution
from shared.normalization import ZScoreNormalizer

Datapoint = tuple[torch.Tensor, int, str]  # (patch tensor, class label, patient id)


class RawDataset(Dataset[Datapoint]):
    """
    Un-augmented data set.
    No geometric augmentation, no class balancing, no per-patch sample weights.

    Contains only global z-score normalization.
    """

    mat_reader: MatReader

    train: bool

    def __init__(
        self,
        mat_reader: MatReader,
        eff_fov_indices: list[int],
        train: bool = True,
        mean_override: torch.Tensor | None = None,
        std_override: torch.Tensor | None = None,
    ) -> None:
        """
        Args:
            mat_reader:      Loaded MatReader instance.
            eff_fov_indices: FOV indices for this split (train or val).
            train:           Does nothing. Left in for interface consistency.
            mean_override:   Pre-computed train-split channel mean (use on val).
            std_override:    Pre-computed train-split channel std (use on val).
        """
        self.mat_reader = mat_reader
        self.eff_fov_indices = eff_fov_indices
        self.train = train
        self.z_normalizer = ZScoreNormalizer(
            mat_reader, eff_fov_indices, mean_override, std_override
        )

        report_class_distribution(mat_reader, eff_fov_indices, train)

    def __len__(self) -> int:
        """Returns number of data points."""
        return len(self.eff_fov_indices)

    def __getitem__(self, idx: int) -> Datapoint:
        eff_idx = self.eff_fov_indices[idx]

        image = self.mat_reader.images[eff_idx]  # (C, H, W)
        image = torch.from_numpy(image).float()
        image = self.z_normalizer.normalize(image)

        class_label = int(self.mat_reader.class_labels[eff_idx])
        patient_id = str(self.mat_reader.patient_ids[eff_idx])

        return image, class_label, patient_id


class LinearSVM(nn.Module):
    """
    Linear SVM classification head on frozen ResNet18 features.
    """

    resnet_model: nn.Module

    def __init__(self, num_classes: int = 4, freeze_all: bool = False) -> None:
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT
        resnet_model = models.resnet18(weights=weights)

        num_features = resnet_model.fc.in_features
        resnet_model.fc = nn.Linear(num_features, num_classes)  # TODO: consider dropout

        for param in resnet_model.parameters():
            param.requires_grad = False
        for param in resnet_model.fc.parameters():
            param.requires_grad = True

        if not freeze_all:
            for param in resnet_model.layer4.parameters():
                param.requires_grad = True

        self.resnet_model = resnet_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.resnet_model(x)


def get_model(num_classes: int = 4, freeze_all: bool = False) -> nn.Module:
    """
    Instantiate an AttentionMIL model.

    Args:
        num_classes: Number of output classes (default 4: Healthy, LGC, HGC, IDC).
        freeze_all:  If True, freeze the entire backbone including layer4.
                     Use with head_only_epochs > 0 for phased unfreezing via
                     OptimizerEngine.maybe_transition_phase().

    Returns:
        nn.Module: The instantiated model, ready for training or inference.
    """
    return LinearSVM(num_classes=num_classes, freeze_all=freeze_all)


def get_loss_fn() -> nn.Module:
    """
    Get the loss function for training.

    Returns:
        nn.Module: The loss function to use during training.
    """
    return nn.MultiMarginLoss()
