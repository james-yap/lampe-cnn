"""
This module defines the entry point for the command-line interface (CLI)
of the LAMPE project using Typer.
It allows users to interact with the application via the command line,
providing options for specifying the architecture and the path to a MAT file.
"""

import os
from datetime import datetime
from enum import Enum
import json

import torch
from torch.utils.data import DataLoader
import typer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from matplotlib import pyplot as plt

from architectures import sliding_window
from shared.mat_reader import MatReader
from shared.early_stopping import EarlyStopping

app = typer.Typer()


class Architecture(str, Enum):
    """
    Model architecture options for training.
    """

    SLIDING_WINDOW = "sliding_window"


@app.command()
def train(
    architecture: Architecture,
    matpath: str,
    num_epochs: int = typer.Option(20, "-e", help="Number of training epochs"),
    n_folds: int = typer.Option(5, "-f", help="Number of folds for cross-validation"),
    batch_size: int = typer.Option(
        32, "-b", help="Batch size for training and validation"
    ),
    lr: float = typer.Option(1e-4, "-l", help="Learning rate for the optimizer"),
    patience: int = typer.Option(5, "-p", help="Patience for early stopping"),
    save_weights: bool = typer.Option(
        False, "-s", help="Whether to save model weights after training"
    ),
):
    """
    Train and evaluate the model based on the specified architecture and MAT file path.
    Automatically creates and saves artifacts (e.g., trained model weights, evaluation metrics) in the 'artifacts' directory.
    Uses StratifiedGroupKFold to ensure balanced representation of classes and groups in training/validation splits, preventing intracore bias.
    """

    start_time = datetime.now()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    mat_reader = MatReader(matpath)

    # random_state is set for reproducibility, but can be removed for more variability in splits across runs
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)

    images, class_labels, patient_ids = (
        mat_reader.images,
        mat_reader.class_labels,
        mat_reader.patient_ids,
    )

    for fold, (train_indices, val_indices) in enumerate(
        sgkf.split(images, class_labels, groups=patient_ids)
    ):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        # TODO: switch case for architectures here based on 'architecture' enum.
        # store hyperparams in report
        # use 'dict' for hyperparams for dynamic architecture
        train_subset = sliding_window.SlidingWindowDataset(
            mat_reader,
            eff_fov_indices=train_indices.tolist(),
            window_size=224,
            stride=96,
        )
        val_subset = sliding_window.SlidingWindowDataset(
            mat_reader, eff_fov_indices=val_indices.tolist(), window_size=224, stride=96
        )

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        num_classes = 4
        model = sliding_window.get_model(num_classes=num_classes).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        early_stopping = EarlyStopping(patience=patience)

        train_losses, val_losses = [], []
        all_preds, all_labels, debug_stratification = [], [], {}

        seen_in_training = set()  # used for data leakage detection

        for epoch in range(num_epochs):
            # train
            model.train()
            running_loss = 0.0
            for patches, labels, _patient_ids in train_loader:
                patches, labels = patches.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(patches)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * patches.size(0)
                for pid in _patient_ids:
                    seen_in_training.add(pid)
            epoch_train_loss = running_loss / len(train_subset)
            train_losses.append(epoch_train_loss)

            # reset predictions and labels to track only last epoch's validation results
            all_preds, all_labels = [], []

            # evaluate
            model.eval()
            val_loss = 0.0
            with torch.no_grad():  # no need to track gradients during validation
                for patches, labels, _patient_ids in val_loader:
                    patches, labels = patches.to(device), labels.to(device)
                    outputs = model(patches)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * patches.size(0)
                    all_preds.append(outputs.cpu())
                    all_labels.append(labels.cpu())
                    for label, pid in zip(labels.cpu().numpy(), _patient_ids):
                        assert (
                            pid not in seen_in_training
                        ), f"Data leakage detected: Patient ID {pid} found in both training and validation sets!"
                        if pid in debug_stratification:
                            if debug_stratification[pid] != label:
                                print(
                                    f"Stratification error: Patient ID {pid} has inconsistent labels across folds (previous: {debug_stratification[pid]}, current: {label})"
                                )
                        else:
                            debug_stratification[pid] = label
            epoch_val_loss = val_loss / len(val_subset)
            val_losses.append(epoch_val_loss)

            print(
                f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f} - Val Loss: {epoch_val_loss:.4f}"
            )

            early_stopping(epoch_val_loss)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        folder_path = ("artifacts", f"{start_time:%m-%d_%H-%M}", f"fold-{fold+1}")
        os.makedirs(os.path.join(*folder_path), exist_ok=True)

        if save_weights:
            torch.save(
                model.state_dict(),
                os.path.join(*folder_path, "model_weights.pth"),
            )

        with open(
            os.path.join(*folder_path, "hyperparams.json"),
            "w",
            encoding="utf-8",
        ) as f:
            hyperparams = {
                "architecture": architecture,
                "matpath": matpath,
                "num_epochs": num_epochs,
                "n_folds": n_folds,
                "batch_size": batch_size,
                "learning_rate": lr,
                "early_stopping_patience": patience,
                "num_patches_per_fov": train_subset.num_patches_per_fov,
            }
            json.dump(hyperparams, f, indent=2)

        if all_preds and all_labels:
            all_labels_np = torch.cat(all_labels).numpy()
            all_preds_cat = torch.cat(all_preds)
            all_probs = torch.softmax(all_preds_cat, dim=1).numpy()

            cm = confusion_matrix(
                all_labels_np,
                all_preds_cat.argmax(dim=1).numpy(),
                labels=list(range(num_classes)),
            )

            cr = classification_report(
                all_labels_np,
                all_preds_cat.argmax(dim=1).numpy(),
                output_dict=False,
                zero_division=0,
            )

            all_labels_bin = label_binarize(
                all_labels_np, classes=list(range(num_classes))
            )

            fig, axes = plt.subplots(2, 2, figsize=(14, 11))
            fig.suptitle(f"Fold {fold + 1} Evaluation", fontsize=14)

            # Loss curve (top-left)
            axes[0, 0].plot(train_losses, label="Train Loss")
            axes[0, 0].plot(val_losses, label="Val Loss")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].set_title("Training and Validation Loss")
            axes[0, 0].legend()

            # Confusion matrix (top-right)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=axes[0, 1], colorbar=False)
            axes[0, 1].set_title("Confusion Matrix")

            # ROC curves (bottom-left)
            for i in range(num_classes):
                if all_labels_bin[:, i].sum() == 0:
                    continue
                fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
                roc_auc = auc(fpr, tpr)
                axes[1, 0].plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")
            axes[1, 0].plot([0, 1], [0, 1], "k--", label="Random")
            axes[1, 0].set_xlabel("False Positive Rate")
            axes[1, 0].set_ylabel("True Positive Rate")
            axes[1, 0].set_title("ROC Curves")
            axes[1, 0].legend()

            # Classification report (bottom-right)
            axes[1, 1].axis("off")
            axes[1, 1].text(
                0.5,
                0.5,  # x and y coordinates set to exactly 50% (the middle)
                cr,
                fontsize=12,  # Bumped up for readability
                family="monospace",
                ha="center",  # Centers the text block horizontally
                va="center",  # Centers the text block vertically
            )
            axes[1, 1].set_title("Classification Report")

            plt.tight_layout()
            plt.savefig(os.path.join(*folder_path, "results.png"), dpi=150)
            plt.close()


@app.command()
def healthcheck():
    """
    Simple health check command to verify that the CLI is working.
    """
    print("LAMPE CLI is up and running!")
