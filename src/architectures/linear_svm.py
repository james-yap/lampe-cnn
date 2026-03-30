"""
Linear SVM as classification head.
Relies on pre-trained feature extractor for non-linearity.
Multi-Class Hinge Loss and L2 Regularization.

Fully frozen ResNet18 backbone as feature extractor, no fine-tuning.
Un-augmented, bad SHG removed.
Works on sparse, imbalanced data as in the original paper (Gagnon et al., 2025).
"""

from shared.constants import CLASS_NAMES
import torch
from typing import cast
from torch import nn
from torch.utils.data import Dataset
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor

from shared.mat_reader import MatReader
from shared.utils import report_class_distribution
from shared.normalization import ZScoreNormalizer

Datapoint = tuple[torch.Tensor, int, str]  # (patch tensor, class label, patient id)


class RawDataset(Dataset[Datapoint]):
    """
    Un-augmented data set.
    No geometric augmentation, no class balancing, no per-patch sample weights.
    """

    mat_reader: MatReader

    train: bool

    def __init__(
        self,
        mat_reader: MatReader,
        eff_fov_indices: list[int],
        train: bool = True,
    ) -> None:
        """
        Args:
            mat_reader:      Loaded MatReader instance.
            eff_fov_indices: FOV indices for this split (train or val).
            train:           Does nothing. Left in for interface consistency.
        """
        self.mat_reader = mat_reader
        self.eff_fov_indices = eff_fov_indices
        self.train = train

        report_class_distribution(mat_reader, eff_fov_indices, train)

    def __len__(self) -> int:
        """Returns number of data points."""
        return len(self.eff_fov_indices)

    def __getitem__(self, idx: int) -> Datapoint:
        eff_idx = self.eff_fov_indices[idx]

        image = self.mat_reader.images[eff_idx]  # (C, H, W)
        image = torch.from_numpy(image).float()

        class_label = int(self.mat_reader.class_labels[eff_idx])
        patient_id = str(self.mat_reader.patient_ids[eff_idx])

        return image, class_label, patient_id


class ZScoreDataset(Dataset[Datapoint]):
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
        image = self.z_normalizer.normalize_and_clip(image)

        class_label = int(self.mat_reader.class_labels[eff_idx])
        patient_id = str(self.mat_reader.patient_ids[eff_idx])

        return image, class_label, patient_id


class LinearSVM(nn.Module):
    """
    Linear SVM classification head on frozen ResNet18 features.
    """

    def __init__(
        self, num_classes: int = len(CLASS_NAMES), freeze_all: bool = False
    ) -> None:
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)

        self.feature_extractor = create_feature_extractor(
            model, return_nodes={"layer2": "features"}
        )

        # phased unfreezing: will be unfrozen later by optimizer
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)  # layer2 outputs 128 channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        features = self.feature_extractor(x)["features"]
        pooled = self.pool(features)
        flattened = torch.flatten(pooled, 1)
        return self.fc(flattened)


class VGG16Model(nn.Module):
    """
    Linear SVM classification head on frozen ResNet18 features.
    """

    model: nn.Module
    classifier_head: nn.Linear

    def __init__(self, num_classes: int = 4, freeze_all: bool = False) -> None:
        super().__init__()

        weights = models.VGG16_Weights.DEFAULT
        model = models.vgg16(weights=weights)
        features = cast(nn.Sequential, model.features)

        for param in features.parameters():
            param.requires_grad = False

        classifier_head = model.classifier[6]
        if not isinstance(classifier_head, nn.Linear):
            raise TypeError("Expected model.classifier[6] to be an nn.Linear layer.")

        in_features = classifier_head.in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(in_features, num_classes),
            nn.Dropout(p=0.5),
        )

        self.model = model
        self.classifier_head = classifier_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)


class ResNet18Model(nn.Module):
    """
    Linear SVM classification head on frozen ResNet18 features.
    """

    resnet_model: nn.Module

    def __init__(self, num_classes: int = 4, freeze_all: bool = False) -> None:
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT
        resnet_model = models.resnet18(weights=weights)

        num_features = resnet_model.fc.in_features
        # resnet_model.fc = nn.Linear(num_features, num_classes)  # TODO: consider dropout
        resnet_model.fc = nn.Sequential(  # type: ignore[assignment]
            nn.Linear(num_features, num_classes),
            nn.Dropout(p=0.5),
        )

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
    return nn.CrossEntropyLoss()
    # return (
    #     nn.MultiMarginLoss()
    # )  # no MPS support: https://github.com/pytorch/pytorch/issues/141287
