"""
Stateful class to compute and apply z-score normalization.
"""

import torch
from shared.mat_reader import MatReader


class ZScoreNormalizer:
    """
    Computes per-channel mean and std from a set of FOVs, then applies z-score normalization to images.
    """

    mat_reader: MatReader
    eff_fov_indices: list[int]
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(
        self,
        mat_reader: MatReader,
        eff_fov_indices: list[int],
        mean_override: torch.Tensor | None = None,
        std_override: torch.Tensor | None = None,
    ) -> None:
        self.mat_reader = mat_reader
        self.eff_fov_indices = eff_fov_indices
        self.mean, self.std = self.compute_per_channel_z_normalization(
            mean_override, std_override
        )

    def compute_per_channel_z_normalization(
        self,
        mean_override: torch.Tensor | None = None,
        std_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes per-channel mean and std for z-score normalization.
        Uses float64 for accumulation to prevent catastrophic precision loss (swamping)
        when summing millions of pixel values across the dataset.
        """
        if (mean_override is None) != (std_override is None):
            raise ValueError(
                "mean_override and std_override must both be provided or both be None."
            )

        if mean_override is not None and std_override is not None:
            return mean_override, std_override

        # Sanity check to prevent DivisionByZero if the dataset split is empty
        if not self.eff_fov_indices:
            raise ValueError(
                "eff_fov_indices cannot be empty when computing statistics."
            )

        num_channels: int = self.mat_reader.get_num_channels()

        # Use float64 to prevent floating-point swamping during massive accumulations
        channel_sums = torch.zeros(num_channels, dtype=torch.float64)
        channel_squared_sums = torch.zeros(num_channels, dtype=torch.float64)
        num_pixels: int = 0

        for fov_idx in self.eff_fov_indices:
            # Cast to float64 immediately to prevent overflow when squaring
            image = torch.from_numpy(self.mat_reader.images[fov_idx]).to(torch.float64)

            channel_sums += image.sum(dim=[1, 2])
            channel_squared_sums += (image**2).sum(dim=[1, 2])
            num_pixels += image.size(1) * image.size(2)

        mean_f64 = channel_sums / num_pixels
        variance_f64 = torch.clamp(
            channel_squared_sums / num_pixels - mean_f64**2, min=0.0
        )
        std_f64 = torch.sqrt(variance_f64)

        # Prevent division by zero for blank channels
        std_f64[std_f64 < 1e-8] = 1.0

        # Cast back to standard float32 for downstream neural network compatibility
        mean = mean_f64.to(torch.float32)
        std = std_f64.to(torch.float32)

        return mean, std

    def normalize(self, image: torch.Tensor, clip: bool = False) -> torch.Tensor:
        """
        Applies z-score normalization to the given image tensor and optionally clips extreme outliers.

        Broadcasting Mechanics:
        -----------------------
        The calculated mean and std are 1D tensors of shape (C,).
        By slicing with `[:, None, None]`, we reshape them to (C, 1, 1).

        PyTorch broadcasting aligns dimensions from right to left:

        Case 1: Single Image (C, H, W)
            Image: (   C,   H,   W)
            Stats: (   C,   1,   1)
            Result: The single stat value for each channel is broadcast across all H and W pixels.

        Case 2: Batched Images (B, C, H, W)
            Image: (B,   C,   H,   W)
            Stats: (     C,   1,   1)
            Result: PyTorch automatically and implicitly assumes a leading batch dimension (B)
                    on the stats tensor because the trailing dimensions (C, 1, 1) align
                    perfectly with the last 3 dimensions of the image batch.
        """
        # Dynamically move mean/std to the incoming image's device (e.g., CPU -> CUDA/MPS)
        # This prevents RuntimeError if the image tensor is already on the GPU
        mean = self.mean.to(image.device)
        std = self.std.to(image.device)

        normalized_image = (image - mean[:, None, None]) / std[:, None, None]

        return torch.clamp(normalized_image, -3.0, 5.0) if clip else normalized_image
