import os
import torch
from torch.utils.data import Dataset, Subset
import scipy.io as sio
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import random_split


file_path = ""
# file_path = "C:/Users/DELL/OneDrive/Desktop/GitHub/Image data-20260326T220833Z-1-001/Image data"


def load_names(mat_file, expected_count):
    # getting the corresponding names file by replacing the suffix of the mat file
    names_file = mat_file.replace("_bulk_data.mat", "_names.txt")
    
    if not os.path.isfile(names_file):
        raise FileNotFoundError(
            f"Names file missing: {names_file}\n"
        )
    # Load names and ensure they match the expected count from the MAT file
    with open(names_file, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f.readlines() if line.strip()]
    if len(names) != expected_count:
        raise ValueError(
            f"Mismatch: {names_file} has {len(names)} names but MAT file has {expected_count} samples"
        )
    
    return names


def _to_group_id(name_value, idx):
    # Convert name to a group ID by taking the first two tokens (e.g., "Patient_01_Sample_01" -> "Patient_01")
    tokens = str(name_value).replace(",", " ").split()
    if len(tokens) >= 2:
        return f"{tokens[0]}_{tokens[1]}"
    if len(tokens) == 1:
        return tokens[0]
    return f"unknown_{idx}"
# further changes to ensure correct group ID extraction and handling of edge cases in names

class ProstateDataset(Dataset):
    def __init__(self, mat_files, class_labels, transform=None):
        self.images = []
        self.labels = []
        self.patient_ids = []
        self.transform = transform

        for mat_file, label in zip(mat_files, class_labels):
            data = sio.loadmat(mat_file)

            # extract modalities
            shg = data[[k for k in data.keys() if "SHG" in k][0]]
            srs1450 = data[[k for k in data.keys() if "1450_onres" in k][0]]
            srs1668 = data[[k for k in data.keys() if "1668_onres" in k][0]]

            N = shg.shape[2]   # number of patches

            # Load names from companion .txt file
            names = load_names(mat_file, N)


            for i in range(N):
                img = np.stack([
                    shg[:, :, i],
                    srs1450[:, :, i],
                    srs1668[:, :, i]
                ], axis=-1)

                img = img.astype(np.uint8)
                self.images.append(img)
                self.labels.append(label)
                self.patient_ids.append(_to_group_id(names[i], i))

    def __len__(self):
        return len(self.images)

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
    mat_files=[
        file_path + "/Healthy_bulk_data.mat",
        file_path + "/LGC_bulk_data.mat",
        file_path + "/HGC_bulk_data.mat",
        file_path + "/IDC_bulk_data.mat"
    ],
    class_labels=[0, 1, 2, 3]
    # healthy - 0, LGC - 1, HGC - 2, TDC - 3
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

print(f"Total samples: {total}")
print(f"Samples for k-fold cross-validation (train+val): {len(train_val_dataset)}")
print(f"Samples in permanent test set: {len(test_dataset)}")
