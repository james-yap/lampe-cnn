import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def EfficientNet(num_classes=4, unfreeze_blocks=0) -> nn.Module:
    # using the smallest version of EfficientNet (B0) for simplicity and to avoid overfitting
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    # Replace the final fully connected layer to match our number of classes (4)
    classifier_layer = model.classifier[1]
    if not isinstance(classifier_layer, nn.Linear):
        raise TypeError("Expected EfficientNet classifier[1] to be nn.Linear")
    model.classifier[1] = nn.Linear(classifier_layer.in_features, num_classes)
    
    # Freezing all backbone feature layers
    for param in model.features.parameters():
        param.requires_grad = False
    # unfreezing the classifier head
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    # Unfreezing last N feature blocks if specified (e.g., unfreeze_blocks=1) 
    # (7 blocks in total for EfficientNet B0) - unfreezing from the end
    if unfreeze_blocks > 0:
        for block in list(model.features.children())[-unfreeze_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
    return model
