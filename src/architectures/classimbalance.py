import torch
from torch.utils.data import Dataset
from shared.mat_reader import MatReader

# (image, class label, patient id)
Datapoint = tuple[torch.Tensor, int, str]


class ClassImbalanceDataset(Dataset[Datapoint]):
    # Dataset wrapper for pre-extracted 3x3 subimages loaded through MatReader.

    def __init__(
        self,
        mat_reader: MatReader,
        eff_indices: list[int],
        train: bool = True,
    ):
        self.mat_reader = mat_reader
        self.eff_indices = eff_indices
        self.train = train

    def __len__(self) -> int:
        return len(self.eff_indices)

    def __getitem__(self, idx: int) -> Datapoint:
        sample_idx = self.eff_indices[idx]

        image = self.mat_reader.images[sample_idx]  # (3, H, W)
        class_label = int(self.mat_reader.class_labels[sample_idx])
        patient_id = str(self.mat_reader.patient_ids[sample_idx])

        image_tensor = torch.from_numpy(image.copy()).float()
        image_tensor = image_tensor / 65535.0

        # train-only geometric augmentation
        if self.train and class_label in [0, 1]:
            k = int(torch.randint(0, 4, (1,)).item())
            image_tensor = torch.rot90(image_tensor, k=k, dims=(1, 2))

        return image_tensor, class_label, patient_id
