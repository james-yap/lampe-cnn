"""
Regularized Architecture: Ordinal K-1 encoding with phased layer unfreezing,
L2 weight decay, and ReduceLROnPlateau learning rate scheduling.

Builds on ordinal.py (same dataset, same K-1 encoding) and adds regularization
support orchestrated by the CLI:

- Weight decay (L2 regularization):
    Adam optimizer weight_decay penalises large weights, slowing memorisation
    of patch-level features that don't generalise to unseen patients.

- Phased layer unfreezing (Direction B):
    Phase 1 (epochs 0 → head_only_epochs − 1): entire backbone frozen, only
      the classification head (~2K parameters) trains.  Forces the model to
      find a good linear combination of ImageNet features before any backbone
      weight is touched.
    Phase 2 (epoch head_only_epochs onward): layer4 is unlocked.  The head is
      already calibrated, so fine-tuning has a stable starting point, reducing
      the risk of catastrophic forgetting and early overfitting.

- ReduceLROnPlateau:
    Monitors val loss after every epoch and halves the learning rate whenever
    it stagnates for 3 epochs (floor: 1e-6).  Prevents large late-epoch steps
    from landing the optimizer outside a generalising basin.

The OrdinalDataset, encode_ordinal, and decode_ordinal helpers are unchanged
from ordinal.py and are re-exported here for use by the CLI.
"""

from torch import nn
from torchvision import models

# Re-export dataset and encoding helpers unchanged from ordinal
from architectures.ordinal import (  # noqa: F401
    OrdinalDatapoint,
    OrdinalDataset,
    CLASS_NAMES,
    encode_ordinal,
    decode_ordinal,
)


def get_model(
    num_classes: int = 4,
    unfreeze_layer3: bool = False,
    freeze_all: bool = False,
) -> nn.Module:
    """
    ResNet18 with pretrained ImageNet weights, partially fine-tuned.
    Outputs K-1 = 3 ordinal logits (one per threshold task).

    Args:
        num_classes (int): Total number of ordinal classes (default: 4).
                           The model head outputs num_classes - 1 = 3 logits.
        unfreeze_layer3 (bool): If True, also unfreeze layer3 in addition to layer4.
        freeze_all (bool): If True, freeze the entire backbone including layer4.
                           Use when head_only_epochs > 0 so the CLI can unfreeze
                           layer4 at the phase boundary rather than from epoch 0.

    Loss: BCEWithLogitsLoss — each of the K-1 outputs is a binary logistic
    regression over a severity threshold.  Loss magnitude is proportional to
    ordinal distance: a Healthy/IDC confusion fires all 3 tasks wrong; an
    LGC/HGC confusion fires only 1.
    """
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Freeze entire network
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer4 unless phased unfreezing is requested
    if not freeze_all:
        for param in model.layer4.parameters():
            param.requires_grad = True

    # Optionally unfreeze layer3 for more fine-tuning capacity
    if unfreeze_layer3 and not freeze_all:
        for param in model.layer3.parameters():
            param.requires_grad = True

    # Replace classification head with K-1 ordinal outputs
    num_ftrs = model.fc.in_features
    num_ordinal_outputs = num_classes - 1  # 3 binary threshold tasks
    model.fc = nn.Sequential(  # type: ignore
        nn.Dropout(0.5), nn.Linear(num_ftrs, num_ordinal_outputs)
    )

    return model
