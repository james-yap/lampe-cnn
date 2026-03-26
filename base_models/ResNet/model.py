import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import random_split

def ResNet18(num_classes=4):
    # loading a pretrained resnet-18 model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # changing the final layer
    model.fc = nn.Linear(in_features=512, out_features=num_classes)

    # freezing every layer/features, except the last fully connected layer 
    # and the last block (layer4) to allow some fine-tuning
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True   # fine-tune these
        else:
            param.requires_grad = False  # freeze others

    return model


