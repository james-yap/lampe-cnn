from matplotlib import transforms
import torch
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import random_split


file_path = ""

class ProstateDataset(Dataset):
    def __init__(self, mat_files, class_labels, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform

        for mat_file, label in zip(mat_files, class_labels):
            data = sio.loadmat(mat_file)

            # extract modalities
            shg = data[[k for k in data.keys() if "SHG" in k][0]]
            srs1450 = data[[k for k in data.keys() if "1450_onres" in k][0]]
            srs1668 = data[[k for k in data.keys() if "1668_onres" in k][0]]

            N = shg.shape[2]   # number of patches


            for i in range(N):
                img = np.stack([
                    shg[:, :, i],
                    srs1450[:, :, i],
                    srs1668[:, :, i]
                ], axis=-1)

                img = img.astype(np.uint8)
                self.images.append(img)
                self.labels.append(label)

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

# dataset spliting into train, validation and test sets
total = len(dataset)
train_size = int(0.7 * total)
val_size = int(0.15 * total)
test_size = total - train_size - val_size 
train_subset, val_subset, test_subset = random_split(dataset,[train_size, val_size, test_size])


# applying tranforms to each set (data augmentation) using the helper class
train_dataset = TransformedSubset(train_subset, transform=train_transforms)
val_dataset = TransformedSubset(val_subset, transform=val_transforms)
test_dataset = TransformedSubset(test_subset, transform=val_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
