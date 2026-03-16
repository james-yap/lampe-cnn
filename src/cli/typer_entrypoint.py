"""
This module defines the entry point for the command-line interface (CLI)
of the LAMPE project using Typer.
It allows users to interact with the application via the command line,
providing options for specifying the architecture and the path to a MAT file.
"""

import os
from enum import Enum

import torch
from torch.utils.data import DataLoader, Subset
import typer
from sklearn.model_selection import StratifiedGroupKFold

from architectures import sliding_window
from shared.mat_reader import MatReader

app = typer.Typer()


class Architecture(str, Enum):
    SLIDING_WINDOW = "sliding_window"


@app.command()
def train(
    architecture: Architecture,
    matpath: str,
    n_folds: int = typer.Option(5, help="Number of folds for cross-validation"),
    batch_size: int = typer.Option(32, help="Batch size for training and validation"),
):
    """
    Train and evaluate the model based on the specified architecture and MAT file path.
    Automatically creates and saves artifacts (e.g., trained model weights, evaluation metrics) in the 'artifacts' directory.
    Uses StratifiedGroupKFold to ensure balanced representation of classes and groups in training/validation splits, preventing intracore bias.
    """

    mat_reader = MatReader(matpath)

    os.makedirs(os.path.join("artifacts", "reports"), exist_ok=True)
    os.makedirs(os.path.join("artifacts", "models"), exist_ok=True)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # random_state is set for reproducibility, but can be removed for more variability in splits across runs
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)

    images, class_labels, patient_ids = (
        mat_reader.images,
        mat_reader.class_labels,
        mat_reader.patient_ids,
    )
    dataset = sliding_window.get_dataset(mat_reader)
    model = sliding_window.get_model(num_classes=4).to(device)

    # TODO: switch case for architectures here based on 'architecture' enum.

    for fold, (train_indices, val_indices) in enumerate(
        sgkf.split(images, class_labels, groups=patient_ids)
    ):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        train_subset = Subset(dataset, train_indices.tolist())
        val_subset = Subset(dataset, val_indices.tolist())
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)


@app.command()
def healthcheck():
    """
    Simple health check command to verify that the CLI is working.
    """
    print("LAMPE CLI is up and running!")
