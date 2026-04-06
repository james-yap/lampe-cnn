"""
Dataset class for binary classification of HGC vs IDC-P images.
"""

import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from shared.normalization import ZScoreNormalizer
from shared.mat_reader import MatReader

Datapoint = tuple[torch.Tensor, int, str]  # (patch tensor, class label, patient id)


class BinaryCEDataset(Dataset[Datapoint]):
    """
    Dataset class for binary classification of HGC vs IDC-P images.
    """

    def __init__(
        self,
        mat_reader: MatReader,
        eff_fov_indices: list[int],
        train: bool = False,
        normalizer_override: ZScoreNormalizer | None = None,
    ) -> None:

        self.mat_reader = mat_reader
        self.eff_fov_indices = eff_fov_indices
        self.train = train

        self.transform = A.Compose(
            [
                # A.ElasticTransform(
                #     alpha=10,
                #     sigma=6,
                #     # alpha_affine=100 * 0.03,
                #     border_mode=cv2.BORDER_REFLECT_101,
                #     p=1.0,
                # ),
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                ToTensorV2(),  # Converts back to (C, H, W) and makes it a Tensor
            ]
        )

        if normalizer_override:
            self.normalizer = normalizer_override
        else:
            self.normalizer = ZScoreNormalizer(mat_reader, eff_fov_indices)

    def __len__(self) -> int:
        return len(self.eff_fov_indices)

    def __getitem__(self, idx: int) -> Datapoint:
        eff_idx = self.eff_fov_indices[idx]

        image = self.mat_reader.images[eff_idx]  # (C, H, W)
        # image = robust_minmax(image, axis=(1, 2)).astype(np.float32)

        if self.train:
            image = image.transpose(1, 2, 0)  # (H, W, C) for albumentations
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(image).float()

        image = self.normalizer.normalize(image)

        class_label = self.mat_reader.class_labels[eff_idx]
        patient_id = self.mat_reader.patient_ids[eff_idx]

        return image, class_label, patient_id
