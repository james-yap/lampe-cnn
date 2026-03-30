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
        """

        mean: torch.Tensor | None = mean_override
        std: torch.Tensor | None = std_override

        if (mean_override is None) != (std_override is None):
            raise ValueError(
                "mean_override and std_override must both be provided or both be None."
            )

        if mean is None and std is None:
            num_channels = self.mat_reader.get_num_channels()
            channel_sums = torch.zeros(num_channels)
            channel_squared_sums = torch.zeros(num_channels)
            num_pixels = 0
            for fov_idx in self.eff_fov_indices:
                image = torch.from_numpy(self.mat_reader.images[fov_idx]).float()
                channel_sums += image.sum(dim=[1, 2])
                channel_squared_sums += (image**2).sum(dim=[1, 2])
                num_pixels += image.size(1) * image.size(2)
            mean = channel_sums / num_pixels
            variance = torch.clamp(channel_squared_sums / num_pixels - mean**2, min=0.0)
            std = torch.sqrt(variance)
            std[std < 1e-8] = 1.0

        assert mean is not None and std is not None
        return mean, std

    def normalize_and_clip(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies z-score normalization to the given image tensor using the computed mean and std.
        Clips the normalized values to a reasonable range to avoid extreme outliers (bright or dark pixels).
        """

        normalized_image = (image - self.mean[:, None, None]) / self.std[:, None, None]
        return torch.clamp(normalized_image, -3.0, 5.0)
