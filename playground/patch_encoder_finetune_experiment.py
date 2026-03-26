"""
Fine-tune an autoencoder encoder for patch classification with:
- class-weighted cross-entropy loss
- weighted random sampling
- optional encoder freezing

Usage
-----
PYTHONPATH=src uv run python playground/patch_encoder_finetune_experiment.py \
  --matpath "lampe_dataset/Full images" \
  --checkpoint "artifacts/03_25-08_10-patch-autoencoder/best_autoencoder.pt" \
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
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

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
    epochs: int = 10
    batch_size: int = 64
    patch_size: int = 64
    stride: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-4
    n_folds: int = 4
    num_workers: int = 0
    max_patches_per_fov: int | None = None
    seed: int = 42
    freeze_encoder: bool = False


class PatchLabeledDataset(Dataset[tuple[torch.Tensor, int, int]]):
    def __init__(
        self,
        mat_reader: MatReader,
        eff_fov_indices: Sequence[int],
        patch_size: int,
        stride: int,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        max_patches_per_fov: int | None = None,
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

        if max_patches_per_fov is not None:
            coords = coords[:max_patches_per_fov]

        self.top_left_coords = coords
        self.num_patches_per_fov = len(coords)

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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        fov_idx = self.eff_fov_indices[idx // self.num_patches_per_fov]
        patch_idx = idx % self.num_patches_per_fov
        y, x = self.top_left_coords[patch_idx]

        patch = self.mat_reader.images[
            fov_idx, :, y : y + self.patch_size, x : x + self.patch_size
        ].astype(np.float32)

        patch = (patch - self.mean) / self.std
        patch_tensor = torch.from_numpy(patch).float()
        class_label = int(self.mat_reader.class_labels[fov_idx])
        fov_id = int(fov_idx)
        return patch_tensor, class_label, fov_id


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
            nn.ConvTranspose2d(
                latent_channels, 64, kernel_size=4, stride=2, padding=1
            ),
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
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder

        if freeze_encoder:
            # fully frozen
            for p in self.encoder.parameters():
                p.requires_grad = False
        else:
            # partial fine-tuning (best setup)
            for p in self.encoder.parameters():
                p.requires_grad = False

            layers = list(self.encoder.children())

            # unfreeze last conv block
            for p in layers[-2].parameters():
                p.requires_grad = True

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
    loader: DataLoader[tuple[torch.Tensor, int, int]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0

    for x, y, _ in loader:
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
def evaluate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    fov_logits = {}
    fov_targets = {}

    for x, y, fov_ids in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        for i in range(len(fov_ids)):
            fov_id = int(fov_ids[i])

            if fov_id not in fov_logits:
                fov_logits[fov_id] = []
                fov_targets[fov_id] = int(y[i].item())

            fov_logits[fov_id].append(logits[i].cpu())

    final_preds = []
    final_targets = []

    for pid in fov_logits:
        stacked = torch.stack(fov_logits[pid]) 
        
        
        k = min(10, stacked.size(0)) # use top 10 patches 
        topk_vals, _ = torch.topk(stacked, k=k, dim=0)
        avg_logits = topk_vals.mean(dim=0)                # [4]

        pred = torch.argmax(avg_logits).item()

        final_preds.append(pred)
        final_targets.append(fov_targets[pid])

    acc = accuracy_score(final_targets, final_preds)

    return 0.0, float(acc), np.array(final_targets), np.array(final_preds)


def make_artifact_dir() -> str:
    t = datetime.now()
    path = os.path.join("artifacts", f"{t:%m_%d-%H_%M}-encoder-finetune")
    os.makedirs(path, exist_ok=True)
    return path


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matpath", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-patches-per-fov", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze encoder instead of fine-tuning it",
    )
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
        freeze_encoder=args.freeze_encoder,
    )


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    artifact_dir = make_artifact_dir()
    with open(
        os.path.join(artifact_dir, "hyperparams.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(asdict(cfg), f, indent=2)

    device = get_device()
    print(f"Using device: {device}")

    print("Loading MatReader...")
    mat_reader = MatReader(cfg.matpath)

    print("Loading autoencoder checkpoint...")
    ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    latent_channels = int(ckpt["config"]["latent_channels"])

    images = mat_reader.images
    class_labels = mat_reader.class_labels
    patient_ids = mat_reader.patient_ids

    sgkf = StratifiedGroupKFold(
        n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed
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
        )
        val_dataset = PatchLabeledDataset(
            mat_reader=mat_reader,
            eff_fov_indices=val_indices.tolist(),
            patch_size=cfg.patch_size,
            stride=cfg.stride,
            mean=train_dataset.mean,
            std=train_dataset.std,
            max_patches_per_fov=cfg.max_patches_per_fov,
        )

        train_labels = mat_reader.class_labels[train_indices]

        class_counts = np.bincount(train_labels, minlength=4).astype(np.float32)
        class_weights = 1.0 / np.sqrt(np.maximum(class_counts, 1.0))
        class_weights = class_weights / class_weights.sum()

        sample_weights = class_weights[train_labels]
        sample_weights = np.repeat(sample_weights, train_dataset.num_patches_per_fov)

        sampler = WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=len(sample_weights),
            replacement=True,
        )

        print(f"Class counts: {class_counts.tolist()}")
        print(f"Class weights: {class_weights.tolist()}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
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
            encoder=autoencoder.encoder,
            latent_channels=latent_channels,
            num_classes=4,
            freeze_encoder=cfg.freeze_encoder,
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        best_val_acc = -1.0
        best_targets: np.ndarray | None = None
        best_preds: np.ndarray | None = None

        for epoch in range(cfg.epochs):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_acc, targets, preds = evaluate(
                model, val_loader, criterion, device
            )

            print(
                f"Epoch {epoch + 1}/{cfg.epochs} - "
                f"Train Loss: {train_loss:.6f} - "
                f"Val Loss: {val_loss:.6f} - "
                f"Val Acc: {val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_targets = targets
                best_preds = preds

        assert best_targets is not None and best_preds is not None
        report = str(
            classification_report(
                best_targets,
                best_preds,
                target_names=["Healthy", "LGC", "HGC", "IDC"],
                digits=4,
                zero_division=0,
            )
        )
        print(f"\nBest fold {fold} classification report:\n{report}")

        with open(
            os.path.join(artifact_dir, f"fold_{fold}_report.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(report)

        fold_results.append({"fold": float(fold), "best_val_acc": best_val_acc})

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
        print(f"Fold {int(r['fold'])}: best val acc = {r['best_val_acc']:.4f}")
    print(f"Mean best val acc: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Artifacts saved to: {artifact_dir}")


if __name__ == "__main__":
    main()