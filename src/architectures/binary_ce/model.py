"""
CustomModel implementation
"""

import torch.utils.model_zoo as model_zoo
import torch
from torch import nn
from torchvision import models
import pretrained_microscopy_models as pmm

from shared.constants import DEVICE


class BinaryCEModel(nn.Module):
    """
    CustomModel implementation
    """

    def __init__(self) -> None:
        super().__init__()

        # weights = models.ResNet18_Weights.DEFAULT
        # model = models.resnet18(weights=weights)

        model = models.resnet50(weights=None)
        url = pmm.util.get_pretrained_microscopynet_url("resnet50", "micronet")
        model.load_state_dict(model_zoo.load_url(url, map_location=DEVICE))

        in_features = model.fc.in_features
        # num_classes = len(CLASS_NAMES)
        # model.fc = nn.Sequential(  # type: ignore[assignment]
        #     nn.BatchNorm1d(in_features), # this is similar to StandardScaler step in SVM
        #     nn.Dropout(p=0.5),
        #     nn.Linear(in_features, num_classes),
        # )

        model.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(32, 1),
        )

        # model.fc = nn.Sequential(
        #     nn.BatchNorm1d(in_features),
        #     nn.Linear(in_features, 512),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(512, 1)
        # )

        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

        self.model = model
        self.fc = model.fc
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        return self.model(x)

    def train(
        self, mode: bool = True
    ) -> "CustomModel":  # `mode` to match PyTorch signature
        # .eval() maps to .train(False)

        super().train(mode)

        if mode:
            # Push the whole backbone (including fc) into eval to freeze BatchNorm stats
            self.model.eval()  # requires_grad is not enough: ".eval()" preserves BatchNorm

            # enables BatchNorm and Dropout of only the classifier
            self.fc.train()

        return self  # pytorch-gradcam calls .eval() and expects an instance


# print(CustomModel()) # inspect architecture
