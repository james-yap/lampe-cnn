"""
Train a patch autoencoder on LAMPE image patches.

Usage
-----
PYTHONPATH=src uv run python playground/patch_autoencoder_experiment.py \
  --matpath "lampe_dataset/Full images" \
  --epochs 20 \
  --batch-size 128 \
  --patch-size 64 \
  --stride 64

Optional:
  --latent-channels 128
  --max-patches-per-fov 64
  --save-reconstructions
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Sequence

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

from shared.mat_reader import MatReader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Config:
    matpath: str
    epochs: int = 20
    batch_size: int = 128
    patch_size: int = 64
    stride: int = 64
    latent_channels: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    n_folds: int = 4
    num_workers: int = 0
    max_patches_per_fov: int | None = None
    seed: int = 42
    save_reconstructions: bool = False
    recon_batches_to_save: int = 1


class PatchAutoencoderDataset(Dataset[torch.Tensor]):
    """
    Unlabeled patch dataset for autoencoder training.

    Returns:
        patch_tensor: (C, H, W)
    """

    def __init__(
        self,
        mat_reader: MatReader,
        eff_fov_indices: Sequence[int],
        patch_size: int,
        stride: int,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        max_patches_per_fov: int | None = None,
        seed: int = 42,
    ) -> None:
        self.mat_reader = mat_reader
        self.eff_fov_indices = list(eff_fov_indices)
        self.patch_size = patch_size
        self.stride = stride
        self.seed = seed

        height, width = mat_reader.get_height_width()
        coords: list[tuple[int, int]] = []
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                coords.append((y, x))

        # Avoid spatial bias from always taking top-left-first patches.
        if max_patches_per_fov is not None and max_patches_per_fov < len(coords):
            rng = random.Random(seed)
            rng.shuffle(coords)
            coords = coords[:max_patches_per_fov]

        self.top_left_coords = coords
        self.num_patches_per_fov = len(coords)

        if self.num_patches_per_fov == 0:
            raise ValueError(
                "No patches were generated. Check patch_size, stride, and image dimensions."
            )

        if mean is None or std is None:
            self.mean, self.std = self._compute_channel_stats()
        else:
            self.mean = mean.astype(np.float32)
            self.std = std.astype(np.float32)

    def _compute_channel_stats(self) -> tuple[np.ndarray, np.ndarray]:
        images = self.mat_reader.images[self.eff_fov_indices]
        mean = images.mean(axis=(0, 2, 3), keepdims=False).astype(np.float32)
        std = images.std(axis=(0, 2, 3), keepdims=False).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        return mean[:, None, None], std[:, None, None]

    def __len__(self) -> int:
        return len(self.eff_fov_indices) * self.num_patches_per_fov

    def __getitem__(self, idx: int) -> torch.Tensor:
        fov_idx = self.eff_fov_indices[idx // self.num_patches_per_fov]
        patch_idx = idx % self.num_patches_per_fov
        y, x = self.top_left_coords[patch_idx]

        patch = self.mat_reader.images[
            fov_idx, :, y : y + self.patch_size, x : x + self.patch_size
        ].astype(np.float32)

        patch = (patch - self.mean) / self.std
        patch_tensor = torch.from_numpy(patch).float()
        return patch_tensor


class ConvPatchAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 3, latent_channels: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def train_one_epoch_autoencoder(
    model: nn.Module,
    loader: DataLoader[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0

    for x in loader:
        x = x.to(device)

        optimizer.zero_grad()
        recon = model(x)
        loss = criterion(recon, x)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_n += bs

    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate_autoencoder(
    model: nn.Module,
    loader: DataLoader[torch.Tensor],
    criterion: nn.Module,
    device: str,
) -> float:
    model.eval()
    total_loss = 0.0
    total_n = 0

    for x in loader:
        x = x.to(device)
        recon = model(x)
        loss = criterion(recon, x)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_n += bs

    return total_loss / max(total_n, 1)


def tensor_to_display_image(x: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized tensor image (C, H, W) into a displayable HWC uint8 image
    by min-max scaling each sample independently.
    """
    arr = x.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))

    min_val = arr.min()
    max_val = arr.max()
    if max_val - min_val < 1e-8:
        arr = np.zeros_like(arr)
    else:
        arr = (arr - min_val) / (max_val - min_val)

    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] > 3:
        arr = arr[:, :, :3]

    return arr


def save_reconstruction_grid(
    model: nn.Module,
    loader: DataLoader[torch.Tensor],
    device: str,
    out_path: str,
    max_items: int = 8,
) -> None:
    """
    Saves a simple side-by-side grid:
    row i: original | reconstruction
    """
    try:
        from PIL import Image
    except ImportError:
        print("PIL not available; skipping reconstruction image save.")
        return

    model.eval()
    batch = next(iter(loader))
    batch = batch.to(device)

    with torch.no_grad():
        recon = model(batch)

    batch = batch[:max_items]
    recon = recon[:max_items]

    originals = [tensor_to_display_image(x) for x in batch]
    reconstructions = [tensor_to_display_image(x) for x in recon]

    h, w, c = originals[0].shape
    n = len(originals)

    canvas = np.zeros((n * h, 2 * w, c), dtype=np.uint8)
    for i in range(n):
        canvas[i * h : (i + 1) * h, 0:w, :] = originals[i]
        canvas[i * h : (i + 1) * h, w : 2 * w, :] = reconstructions[i]

    image = Image.fromarray(canvas)
    image.save(out_path)


def make_artifact_dir() -> str:
    t = datetime.now()
    path = os.path.join("artifacts", f"{t:%m_%d-%H_%M}-patch-autoencoder")
    os.makedirs(path, exist_ok=True)
    return path


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matpath", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--latent-channels", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-patches-per-fov", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-reconstructions", action="store_true")
    parser.add_argument("--recon-batches-to-save", type=int, default=1)
    args = parser.parse_args()

    return Config(
        matpath=args.matpath,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        stride=args.stride,
        latent_channels=args.latent_channels,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_folds=args.n_folds,
        num_workers=args.num_workers,
        max_patches_per_fov=args.max_patches_per_fov,
        seed=args.seed,
        save_reconstructions=args.save_reconstructions,
        recon_batches_to_save=args.recon_batches_to_save,
    )


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    artifact_dir = make_artifact_dir()
    with open(os.path.join(artifact_dir, "hyperparams.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    device = get_device()
    print(f"Using device: {device}")

    print("Loading MatReader...")
    mat_reader = MatReader(cfg.matpath)

    images = mat_reader.images
    class_labels = mat_reader.class_labels
    patient_ids = mat_reader.patient_ids

    sgkf = StratifiedGroupKFold(
        n_splits=cfg.n_folds,
        shuffle=True,
        random_state=cfg.seed,
    )

    fold_results: list[dict[str, float]] = []

    best_global_val_loss = math.inf
    best_global_checkpoint_path = os.path.join(artifact_dir, "best_autoencoder.pt")

    for fold, (train_indices, val_indices) in enumerate(
        sgkf.split(images, class_labels, groups=patient_ids),
        start=1,
    ):
        print(f"\n--- Fold {fold}/{cfg.n_folds} ---")

        train_dataset = PatchAutoencoderDataset(
            mat_reader=mat_reader,
            eff_fov_indices=train_indices.tolist(),
            patch_size=cfg.patch_size,
            stride=cfg.stride,
            max_patches_per_fov=cfg.max_patches_per_fov,
            seed=cfg.seed + fold,
        )
        val_dataset = PatchAutoencoderDataset(
            mat_reader=mat_reader,
            eff_fov_indices=val_indices.tolist(),
            patch_size=cfg.patch_size,
            stride=cfg.stride,
            mean=train_dataset.mean,
            std=train_dataset.std,
            max_patches_per_fov=cfg.max_patches_per_fov,
            seed=cfg.seed + fold,
        )

        print(
            f"Train patches: {len(train_dataset)} | "
            f"Val patches: {len(val_dataset)} | "
            f"Patches/FOV: {train_dataset.num_patches_per_fov}"
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )

        model = ConvPatchAutoencoder(
            in_channels=mat_reader.get_num_channels(),
            latent_channels=cfg.latent_channels,
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        best_fold_val_loss = math.inf
        fold_history: list[dict[str, float]] = []

        for epoch in range(cfg.epochs):
            train_loss = train_one_epoch_autoencoder(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )
            val_loss = evaluate_autoencoder(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )

            fold_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
            )

            print(
                f"Epoch {epoch + 1}/{cfg.epochs} - "
                f"Train Loss: {train_loss:.6f} - "
                f"Val Loss: {val_loss:.6f}"
            )

            if val_loss < best_fold_val_loss:
                best_fold_val_loss = val_loss

                fold_ckpt_path = os.path.join(artifact_dir, f"fold_{fold}_best_autoencoder.pt")
                torch.save(
                    {
                        "fold": fold,
                        "config": asdict(cfg),
                        "model_state_dict": model.state_dict(),
                        "train_mean": train_dataset.mean,
                        "train_std": train_dataset.std,
                        "in_channels": mat_reader.get_num_channels(),
                        "latent_channels": cfg.latent_channels,
                        "best_val_loss": best_fold_val_loss,
                    },
                    fold_ckpt_path,
                )

                if best_fold_val_loss < best_global_val_loss:
                    best_global_val_loss = best_fold_val_loss
                    torch.save(
                        {
                            "fold": fold,
                            "config": asdict(cfg),
                            "model_state_dict": model.state_dict(),
                            "train_mean": train_dataset.mean,
                            "train_std": train_dataset.std,
                            "in_channels": mat_reader.get_num_channels(),
                            "latent_channels": cfg.latent_channels,
                            "best_val_loss": best_fold_val_loss,
                        },
                        best_global_checkpoint_path,
                    )

                    if cfg.save_reconstructions:
                        recon_path = os.path.join(
                            artifact_dir,
                            f"best_global_fold_{fold}_reconstructions.png",
                        )
                        save_reconstruction_grid(
                            model=model,
                            loader=val_loader,
                            device=device,
                            out_path=recon_path,
                            max_items=8,
                        )

        history_path = os.path.join(artifact_dir, f"fold_{fold}_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(fold_history, f, indent=2)

        fold_results.append(
            {
                "fold": fold,
                "best_val_loss": best_fold_val_loss,
            }
        )

    mean_val_loss = float(np.mean([r["best_val_loss"] for r in fold_results]))
    std_val_loss = float(np.std([r["best_val_loss"] for r in fold_results]))

    summary = {
        "fold_results": fold_results,
        "mean_best_val_loss": mean_val_loss,
        "std_best_val_loss": std_val_loss,
        "best_global_val_loss": best_global_val_loss,
        "best_global_checkpoint": best_global_checkpoint_path,
    }

    with open(os.path.join(artifact_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    for r in fold_results:
        print(f"Fold {r['fold']}: best val loss = {r['best_val_loss']:.6f}")
    print(f"Mean best val loss: {mean_val_loss:.6f} ± {std_val_loss:.6f}")
    print(f"Best global val loss: {best_global_val_loss:.6f}")
    print(f"Artifacts saved to: {artifact_dir}")


if __name__ == "__main__":
    main()
