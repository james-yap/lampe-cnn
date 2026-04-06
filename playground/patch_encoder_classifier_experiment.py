"""
Train a small classifier on top of a frozen patch-autoencoder encoder
and evaluate at the FOV level using mean-logit aggregation.

Usage
-----
PYTHONPATH=src uv run python playground/patch_encoder_classifier_experiment.py \
  --matpath "lampe_dataset/Full images" \
  --checkpoint "artifacts/04_05-20_33-patch-autoencoder/best_autoencoder.pt" \
  --epochs 10 \
  --batch-size 64 \
  --patch-size 64 \
  --stride 64
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from shared.constants import CLASS_NAMES

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
    checkpoint: str
    epochs: int = 30
    batch_size: int = 64
    patch_size: int = 64
    stride: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    n_folds: int = 4
    num_workers: int = 0
    max_patches_per_fov: int | None = None
    seed: int = 42


class PatchLabeledDataset(Dataset[tuple[torch.Tensor, int, int, str]]):
    """
    Patch dataset that uses parent FOV label as patch label.

    Returns:
        patch_tensor: (C, H, W)
        class_label: int
        fov_idx: int
        patient_id: str
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

        height, width = mat_reader.get_height_width()
        coords: list[tuple[int, int]] = []
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                coords.append((y, x))

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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int, str]:
        fov_idx = self.eff_fov_indices[idx // self.num_patches_per_fov]
        patch_idx = idx % self.num_patches_per_fov
        y, x = self.top_left_coords[patch_idx]

        patch = self.mat_reader.images[
            fov_idx, :, y : y + self.patch_size, x : x + self.patch_size
        ].astype(np.float32)

        patch = (patch - self.mean) / self.std
        patch_tensor = torch.from_numpy(patch).float()
        class_label = int(self.mat_reader.class_labels[fov_idx])
        patient_id = str(self.mat_reader.patient_ids[fov_idx])
        return patch_tensor, class_label, int(fov_idx), patient_id


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


class EncoderClassifier(nn.Module):
    def __init__(
        self, 
        encoder: nn.Module, 
        latent_channels: int = 128,
        num_classes: int = 4,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = not freeze_encoder

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        feats = self.pool(feats)
        return self.classifier(feats)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int, int, str]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0

    for x, y, _, _ in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_n += bs

    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate_fov(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int, int, str]],
    criterion: nn.Module,
    device: str,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_patch_loss = 0.0
    total_patch_n = 0

    fov_logits: dict[int, list[np.ndarray]] = defaultdict(list)
    fov_targets: dict[int, int] = {}

    for x, y, fov_idx, _ in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        total_patch_loss += loss.item() * bs
        total_patch_n += bs

        logits_np = logits.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        fov_idx_np = np.asarray(fov_idx)

        for i in range(bs):
            this_fov = int(fov_idx_np[i])
            fov_logits[this_fov].append(logits_np[i])
            if this_fov in fov_targets:
                if fov_targets[this_fov] != int(y_np[i]):
                    raise ValueError(f"Inconsistent labels found for FOV {this_fov}.")
            else:
                fov_targets[this_fov] = int(y_np[i])

    all_targets: list[int] = []
    all_preds: list[int] = []

    for this_fov in sorted(fov_logits.keys()):
        mean_logits = np.mean(np.stack(fov_logits[this_fov], axis=0), axis=0)
        pred = int(np.argmax(mean_logits))
        target = fov_targets[this_fov]

        all_targets.append(target)
        all_preds.append(pred)

    acc = accuracy_score(all_targets, all_preds)

    return (
        total_patch_loss / max(total_patch_n, 1),
        float(acc),
        np.array(all_targets),
        np.array(all_preds),
    )


def make_artifact_dir() -> str:
    t = datetime.now()
    path = os.path.join("artifacts", f"{t:%m_%d-%H_%M}-encoder-classifier-fov")
    os.makedirs(path, exist_ok=True)
    return path


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matpath", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-patches-per-fov", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return Config(
        matpath=args.matpath,
        checkpoint=args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        stride=args.stride,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_folds=args.n_folds,
        num_workers=args.num_workers,
        max_patches_per_fov=args.max_patches_per_fov,
        seed=args.seed,
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

    print("Loading autoencoder checkpoint...")
    ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)

    if "latent_channels" in ckpt:
        latent_channels = int(ckpt["latent_channels"])
    else:
        latent_channels = int(ckpt["config"]["latent_channels"])

    images = mat_reader.images
    class_labels = mat_reader.class_labels
    patient_ids = mat_reader.patient_ids

    sgkf = StratifiedGroupKFold(
        n_splits=cfg.n_folds,
        shuffle=True,
        random_state=cfg.seed,
    )

    fold_results: list[dict[str, float]] = []

    for fold, (train_indices, val_indices) in enumerate(
        sgkf.split(images, class_labels, groups=patient_ids),
        start=1,
    ):
        print(f"\n--- Fold {fold}/{cfg.n_folds} ---")

        train_dataset = PatchLabeledDataset(
            mat_reader=mat_reader,
            eff_fov_indices=train_indices.tolist(),
            patch_size=cfg.patch_size,
            stride=cfg.stride,
            max_patches_per_fov=cfg.max_patches_per_fov,
            seed=cfg.seed + fold,
        )
        val_dataset = PatchLabeledDataset(
            mat_reader=mat_reader,
            eff_fov_indices=val_indices.tolist(),
            patch_size=cfg.patch_size,
            stride=cfg.stride,
            mean=train_dataset.mean,
            std=train_dataset.std,
            max_patches_per_fov=cfg.max_patches_per_fov,
            seed=cfg.seed + fold,
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

        autoencoder = ConvPatchAutoencoder(
            in_channels=mat_reader.get_num_channels(),
            latent_channels=latent_channels,
        )
        autoencoder.load_state_dict(ckpt["model_state_dict"])

        model = EncoderClassifier(
            encoder = autoencoder.encoder,
            latent_channels=latent_channels,
            num_classes=len(CLASS_NAMES),
            freeze_encoder=False,
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        best_val_acc = -1.0
        best_targets: np.ndarray | None = None
        best_preds: np.ndarray | None = None
        
        patience = 5
        epochs_no_improve = 0
        for epoch in range(cfg.epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, targets, preds = evaluate_fov(model, val_loader, criterion, device)

            print(
                f"Epoch {epoch + 1}/{cfg.epochs} - "
                f"Train Loss: {train_loss:.6f} - "
                f"Val Patch Loss: {val_loss:.6f} - "
                f"Val FOV Acc: {val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_targets = targets
                best_preds = preds
                best_epoch = epoch + 1
                epochs_no_improve = 0
                
                torch.save(
                    {
                        "fold": fold,
                        "epoch": best_epoch,
                        "best_val_acc": best_val_acc,
                        "model_state_dict": model.state_dict(),
                        "config": asdict(cfg),
                        "class_names": CLASS_NAMES,
                    },
                    os.path.join(artifact_dir, f"fold_{fold}_best.pt"),
                )
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        assert best_targets is not None and best_preds is not None
        report = str(
            classification_report(
                best_targets,
                best_preds,
                labels = list(range(len(CLASS_NAMES))),
                target_names=CLASS_NAMES,
                digits=4,
                zero_division=0,
            )
        )
        print(f"\nBest fold {fold} FOV classification report:\n{report}")

        with open(os.path.join(artifact_dir, f"fold_{fold}_report.txt"), "w", encoding="utf-8") as f:
            f.write(report)

        fold_results.append(
            {
                "fold": float(fold), 
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                }
            )

    mean_acc = float(np.mean([r["best_val_acc"] for r in fold_results]))
    std_acc = float(np.std([r["best_val_acc"] for r in fold_results]))

    summary = {
        "fold_results": fold_results,
        "mean_best_val_acc": mean_acc,
        "std_best_val_acc": std_acc,
    }

    with open(os.path.join(artifact_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    for r in fold_results:
        epoch_str = f"(epoch {int(r['best_epoch'])})" if 'best_epoch' in r else ""
        print(
                f"Fold {int(r['fold'])}: best val FOV acc = {r['best_val_acc']:.4f} {epoch_str}"
            )
    print(f"Mean best val FOV acc: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Artifacts saved to: {artifact_dir}")


if __name__ == "__main__":
    main()
