import sys
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import random_split


sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.shared.mat_reader import MatReader


file_path = ""
# file path with all the .mat files and the companion .txt files for patient IDs and patch names.
file_path = "C:/Users/DELL/OneDrive/Desktop/GitHub/full_images/Full images/"

# class to load the dataset with the MatReader and convert it into a PyTorch Dataset
class ProstateDataset(Dataset):
    def __init__(self, matpath, transform=None):
        self.transform = transform

        mat_reader = MatReader(matpath)

        # MatReader returns (n, c, w, h). PIL expects HWC.
        images_hwc = np.transpose(mat_reader.images, (0, 2, 3, 1))
        self.images = np.clip(images_hwc, 0, 255).astype(np.uint8)
        self.labels = mat_reader.class_labels.astype(np.int64)
        # Patient IDs are used for grouping in StratifiedGroupKFold to prevent data leakage 
        # between training and validation sets
        self.patient_ids = mat_reader.patient_ids

    # the length of the dataset is the number of samples (n)
    def __len__(self):
        return len(self.images)

    #  to get an item from the dataset
    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        label = self.labels[idx]

        # The transform is now applied within this method after the split
        if self.transform:
            img = self.transform(img)

        return img, label

# Helper class to apply transforms to subsets
class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.subset[index] # Get item from the original dataset through the subset
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.subset)


dataset = ProstateDataset(
    matpath=file_path
    # transform=train_transform # Transform will be applied after split
)

torch.manual_seed(45)

# tranforms for input images
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Dataset splitting into a permanent test set and a combined train/validation set
total = len(dataset)
test_split_size = int(0.15 * total) # Allocate 15% for the final test set
train_val_size = total - test_split_size

# Splitting into train_val_dataset and test_subset
train_val_dataset, test_subset_raw = random_split(dataset, [train_val_size, test_split_size])

# Extracting labels and group IDs for the train_val_dataset to use in StratifiedGroupKFold
train_val_indices = np.array(train_val_dataset.indices)
train_val_labels = np.array(dataset.labels)[train_val_indices]
train_val_groups = np.array(dataset.patient_ids)[train_val_indices]

# Applying transformations to the permanent test set
test_dataset = TransformedSubset(test_subset_raw, transform=val_transforms)

# Creating a DataLoader for the permanent test set
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

