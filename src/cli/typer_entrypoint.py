"""
This module defines the entry point for the command-line interface (CLI)
of the LAMPE project using Typer.
It allows users to interact with the application via the command line,
providing options for specifying the architecture and the path to a MAT file.
"""

# built-in
from enum import Enum

# third-party
import typer

app = typer.Typer()

VERBOSE_MODE = False
NUM_CLASSES = 4


class Architecture(str, Enum):
    """
    Model architecture options for training.
    """

    SLIDING_WINDOW = "sliding_window"
    STANDARDIZED = "standardized"
    CLASS_BALANCED = "class_balanced"
    ORDINAL = "ordinal"
    REGULARIZED = "regularized"
    CONTINUOUS_AUG = "continuous_aug"
    MIL = "mil"


@app.command()
def train(
    architecture: Architecture,
    matpath: str,
    num_epochs: int = typer.Option(60, "-e", help="Number of training epochs"),
    n_folds: int = typer.Option(4, "-f", help="Number of folds for cross-validation"),
    batch_size: int = typer.Option(
        32, "-b", help="Batch size for training and validation"
    ),
    lr: float = typer.Option(1e-4, "-l", help="Learning rate for the optimizer"),
    patience: int = typer.Option(10, "-p", help="Patience for early stopping"),
    save_weights: bool = typer.Option(
        False, "-s", help="Whether to save model weights after training"
    ),
    weight_decay: float = typer.Option(
        1e-4, "-w", help="L2 weight decay for Adam optimizer (REGULARIZED only)"
    ),
    head_only_epochs: int = typer.Option(
        50,
        "-H",
        help="Epochs to train head-only before unfreezing layer4 (REGULARIZED only)",
    ),
):
    """
    Train and evaluate the model based on the specified architecture and MAT file path.
    Automatically creates and saves artifacts
    (e.g., trained model weights, evaluation metrics) in the 'artifacts' directory.
    Uses StratifiedGroupKFold to ensure balanced representation of classes
    and groups in training/validation splits, preventing intracore bias.
    """
    print(f"Starting training with architecture: {architecture.value}.")
    print("Initializing PyTorch engine...\n")

    # pylint: disable=import-outside-toplevel
    # to allow lazy loading (speed up initial CLI response time)

    # built-in
    import os
    from datetime import datetime
    import json

    # third-party
    import torch
    from torch.utils.data import DataLoader
    from sklearn.model_selection import StratifiedGroupKFold

    # package
    from architectures import (
        sliding_window,
        standardized,
        class_balanced,
        ordinal,
        regularized,
        continuous_aug,
        mil,
    )
    from torch.utils.data import WeightedRandomSampler
    from shared.mat_reader import MatReader
    from shared.early_stopping import EarlyStopping
    from shared.report import FoldReporter
    from shared.optimizer_engine import OptimizerEngine

    mat_reader = MatReader(matpath)

    hyperparams = {
        "architecture": architecture,
        "matpath": matpath,
        "num_epochs": num_epochs,
        "n_folds": n_folds,
        "batch_size": batch_size,
        "learning_rate": lr,
        "early_stopping_patience": patience,
        "sliding_factor": 5,
        "weight_decay": weight_decay,
        "head_only_epochs": head_only_epochs,
        "lr_scheduler": (
            "ReduceLROnPlateau"
            if architecture
            in (
                Architecture.REGULARIZED,
                Architecture.CONTINUOUS_AUG,
                Architecture.MIL,
            )
            else "none"
        ),
    }

    start_time = datetime.now()
    artifact_folder_path = os.path.join(
        "artifacts",
        f"{start_time:%m_%d-%H_%M}-{hyperparams['architecture'].value}",
    )
    os.makedirs(artifact_folder_path, exist_ok=True)

    stub_dataset = sliding_window.SlidingWindowDataset(
        mat_reader,
        eff_fov_indices=[
            0
        ],  # use only the first FOV to compute num_patches_per_fov for hyperparameter logging
        factor=hyperparams["sliding_factor"],
    )
    hyperparams["num_patches_per_fov"] = stub_dataset.num_patches_per_fov

    with open(
        os.path.join(artifact_folder_path, "hyperparams.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(hyperparams, f, indent=2)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # random_state is set for reproducibility,
    # but can be removed for more variability in splits across runs
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)

    images, class_labels, patient_ids = (
        mat_reader.images,
        mat_reader.class_labels,
        mat_reader.patient_ids,
    )

    reporter = FoldReporter(
        num_classes=NUM_CLASSES,
        class_names=["Healthy", "LGC", "HGC", "IDC"],
    )

    for fold, (train_indices, val_indices) in enumerate(
        sgkf.split(images, class_labels, groups=patient_ids)
    ):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        if architecture == Architecture.SLIDING_WINDOW:
            train_subset = sliding_window.SlidingWindowDataset(
                mat_reader,
                eff_fov_indices=train_indices.tolist(),
                factor=hyperparams["sliding_factor"],
            )
            val_subset = sliding_window.SlidingWindowDataset(
                mat_reader,
                eff_fov_indices=val_indices.tolist(),
                factor=hyperparams["sliding_factor"],
            )
            model = sliding_window.get_model(num_classes=NUM_CLASSES).to(device)
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        elif architecture == Architecture.STANDARDIZED:
            train_subset = standardized.StandardizedDataset(
                mat_reader,
                eff_fov_indices=train_indices.tolist(),
                factor=hyperparams["sliding_factor"],
            )
            val_subset = standardized.StandardizedDataset(
                mat_reader,
                eff_fov_indices=val_indices.tolist(),
                factor=hyperparams["sliding_factor"],
                mean_override=train_subset.mean,  # prevent data leakage by using train stats
                std_override=train_subset.std,
            )
            model = standardized.get_model(num_classes=NUM_CLASSES).to(device)
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        elif architecture == Architecture.CLASS_BALANCED:
            cb_train = class_balanced.ClassBalancedDataset(
                mat_reader,
                eff_fov_indices=train_indices.tolist(),
                factor=hyperparams["sliding_factor"],
                train=True,
            )
            val_subset = class_balanced.ClassBalancedDataset(
                mat_reader,
                eff_fov_indices=val_indices.tolist(),
                factor=hyperparams["sliding_factor"],
                train=False,
                mean_override=cb_train.mean,  # prevent data leakage by using train stats
                std_override=cb_train.std,
            )
            train_subset = cb_train
            model = class_balanced.get_model(num_classes=NUM_CLASSES).to(device)
            sampler = WeightedRandomSampler(
                weights=cb_train.sample_weights.tolist(),
                num_samples=len(cb_train),
                replacement=True,
            )
            train_loader = DataLoader(
                train_subset, batch_size=batch_size, sampler=sampler
            )
        elif architecture == Architecture.ORDINAL:
            ord_train = ordinal.OrdinalDataset(
                mat_reader,
                eff_fov_indices=train_indices.tolist(),
                factor=hyperparams["sliding_factor"],
                train=True,
            )
            val_subset = ordinal.OrdinalDataset(
                mat_reader,
                eff_fov_indices=val_indices.tolist(),
                factor=hyperparams["sliding_factor"],
                train=False,
                mean_override=ord_train.mean,  # prevent data leakage by using train stats
                std_override=ord_train.std,
            )
            train_subset = ord_train
            model = ordinal.get_model(num_classes=NUM_CLASSES).to(device)
            sampler = WeightedRandomSampler(
                weights=ord_train.sample_weights.tolist(),
                num_samples=len(ord_train),
                replacement=True,
            )
            train_loader = DataLoader(
                train_subset, batch_size=batch_size, sampler=sampler
            )
        elif architecture == Architecture.REGULARIZED:
            reg_train = regularized.OrdinalDataset(
                mat_reader,
                eff_fov_indices=train_indices.tolist(),
                factor=hyperparams["sliding_factor"],
                train=True,
            )
            val_subset = regularized.OrdinalDataset(
                mat_reader,
                eff_fov_indices=val_indices.tolist(),
                factor=hyperparams["sliding_factor"],
                train=False,
                mean_override=reg_train.mean,  # prevent data leakage by using train stats
                std_override=reg_train.std,
            )
            train_subset = reg_train
            model = regularized.get_model(
                num_classes=NUM_CLASSES, freeze_all=(head_only_epochs > 0)
            ).to(device)
            sampler = WeightedRandomSampler(
                weights=reg_train.sample_weights.tolist(),
                num_samples=len(reg_train),
                replacement=True,
            )
            train_loader = DataLoader(
                train_subset, batch_size=batch_size, sampler=sampler
            )
        elif architecture == Architecture.CONTINUOUS_AUG:
            ca_train = continuous_aug.ContinuousAugDataset(
                mat_reader,
                eff_fov_indices=train_indices.tolist(),
                factor=hyperparams["sliding_factor"],
                train=True,
            )
            val_subset = continuous_aug.ContinuousAugDataset(
                mat_reader,
                eff_fov_indices=val_indices.tolist(),
                factor=hyperparams["sliding_factor"],
                train=False,
                mean_override=ca_train.mean,
                std_override=ca_train.std,
            )
            train_subset = ca_train
            model = continuous_aug.get_model(
                num_classes=NUM_CLASSES, freeze_all=(head_only_epochs > 0)
            ).to(device)
            sampler = WeightedRandomSampler(
                weights=ca_train.sample_weights.tolist(),
                num_samples=len(ca_train),
                replacement=True,
            )
            train_loader = DataLoader(
                train_subset, batch_size=batch_size, sampler=sampler
            )
        elif architecture == Architecture.MIL:
            mil_train = mil.MILDataset(
                mat_reader,
                eff_fov_indices=train_indices.tolist(),
                factor=hyperparams["sliding_factor"],
                train=True,
            )
            val_subset = mil.MILDataset(
                mat_reader,
                eff_fov_indices=val_indices.tolist(),
                factor=hyperparams["sliding_factor"],
                train=False,
                mean_override=mil_train.mean,
                std_override=mil_train.std,
            )
            train_subset = mil_train
            model = mil.get_model(
                num_classes=NUM_CLASSES, freeze_all=(head_only_epochs > 0)
            ).to(device)
            sampler = WeightedRandomSampler(
                weights=mil_train.sample_weights.tolist(),
                num_samples=len(mil_train),
                replacement=True,
            )
            train_loader = DataLoader(
                train_subset, batch_size=batch_size, sampler=sampler
            )
        else:
            raise NotImplementedError(f"Architecture {architecture} not implemented.")

        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        engine = OptimizerEngine.for_architecture(
            architecture=architecture.value,
            model=model,
            lr=lr,
            weight_decay=weight_decay,
            head_only_epochs=head_only_epochs,
        )
        early_stopping = EarlyStopping(patience=patience)

        train_losses, val_losses = [], []
        all_preds, all_labels, debug_stratification = [], [], {}
        train_preds_last: list[torch.Tensor] = []
        train_labels_last: list[torch.Tensor] = []

        seen_in_training = set()  # used for data leakage detection

        for epoch in range(num_epochs):
            if engine.maybe_transition_phase(epoch):
                print(f"  [Phase 2] Unfreezing layer4 at epoch {epoch + 1}")

            # train
            model.train()
            running_loss = 0.0
            train_preds_last, train_labels_last = [], []
            for batch in train_loader:
                if architecture == Architecture.MIL:
                    bags, int_class_labels, _patient_ids = batch
                    bags = bags.to(device)
                    targets = int_class_labels.to(device)
                    engine.optimizer.zero_grad()
                    mil_out: tuple[torch.Tensor, torch.Tensor] = model(bags)  # type: ignore[assignment]
                    mil_logits, _mil_attn = mil_out
                    outputs = mil_logits
                    loss = engine.criterion(outputs, targets)
                    loss.backward()
                    engine.optimizer.step()
                    running_loss += loss.item() * bags.size(0)
                elif architecture in (
                    Architecture.ORDINAL,
                    Architecture.REGULARIZED,
                    Architecture.CONTINUOUS_AUG,
                ):
                    patches, ordinal_targets, int_class_labels, _patient_ids = batch
                    patches = patches.to(device)
                    targets = ordinal_targets.to(device)  # (batch, K-1) float
                    engine.optimizer.zero_grad()
                    outputs = model(patches)
                    loss = engine.criterion(outputs, targets)
                    loss.backward()
                    engine.optimizer.step()
                    running_loss += loss.item() * patches.size(0)
                else:
                    patches, int_class_labels, _patient_ids = batch
                    patches = patches.to(device)
                    targets = int_class_labels.to(device)
                    engine.optimizer.zero_grad()
                    outputs = model(patches)
                    loss = engine.criterion(outputs, targets)
                    loss.backward()
                    engine.optimizer.step()
                    running_loss += loss.item() * patches.size(0)
                # collect for train confusion matrix (last epoch's data used at report time)
                train_preds_last.append(outputs.detach().cpu())
                train_labels_last.append(int_class_labels.cpu())
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
                for batch in val_loader:
                    if architecture == Architecture.MIL:
                        bags, int_class_labels, _patient_ids = batch
                        bags = bags.to(device)
                        targets = int_class_labels.to(device)
                        mil_val_out: tuple[torch.Tensor, torch.Tensor] = model(bags)  # type: ignore[assignment]
                        mil_val_logits, _mil_val_attn = mil_val_out
                        outputs = mil_val_logits
                        loss = engine.criterion(outputs, targets)
                        val_loss += loss.item() * bags.size(0)
                    elif architecture in (
                        Architecture.ORDINAL,
                        Architecture.REGULARIZED,
                        Architecture.CONTINUOUS_AUG,
                    ):
                        patches, ordinal_targets, int_class_labels, _patient_ids = batch
                        patches = patches.to(device)
                        targets = ordinal_targets.to(device)  # (batch, K-1) float
                        outputs = model(patches)
                        loss = engine.criterion(outputs, targets)
                        val_loss += loss.item() * patches.size(0)
                    else:
                        patches, int_class_labels, _patient_ids = batch
                        patches = patches.to(device)
                        targets = int_class_labels.to(device)
                        outputs = model(patches)
                        loss = engine.criterion(outputs, targets)
                        val_loss += loss.item() * patches.size(0)
                    all_preds.append(outputs.cpu())
                    all_labels.append(int_class_labels.cpu())
                    for label, pid in zip(int_class_labels.numpy(), _patient_ids):
                        assert pid not in seen_in_training, (
                            "Data leakage detected: "
                            f"Patient ID {pid} found in both training and validation sets!"
                        )
                        if pid in debug_stratification:
                            if debug_stratification[pid] != label and VERBOSE_MODE:
                                print(
                                    (
                                        "Stratification error: "
                                        f"Patient ID {pid} has inconsistent labels across folds"
                                        f"(previous: {debug_stratification[pid]}, current: {label})"
                                    )
                                )
                        else:
                            debug_stratification[pid] = label
            epoch_val_loss = val_loss / len(val_subset)
            val_losses.append(epoch_val_loss)

            print(
                (
                    f"Epoch {epoch + 1}/{num_epochs}"
                    f"- Train Loss: {epoch_train_loss:.4f}"
                    f"- Val Loss: {epoch_val_loss:.4f}"
                )
            )

            early_stopping(epoch_val_loss)
            engine.step_scheduler(epoch_val_loss)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        path_with_kfold = os.path.join(artifact_folder_path, f"fold-{fold+1}")
        os.makedirs(path_with_kfold, exist_ok=True)

        if save_weights:
            torch.save(
                model.state_dict(),
                os.path.join(path_with_kfold, "model_weights.pth"),
            )

        if all_preds and all_labels:
            reporter.save(
                fold=fold + 1,
                output_dir=path_with_kfold,
                train_losses=train_losses,
                val_losses=val_losses,
                all_preds=torch.cat(all_preds),
                all_labels=torch.cat(all_labels),
                train_preds=torch.cat(train_preds_last) if train_preds_last else None,
                train_labels=(
                    torch.cat(train_labels_last) if train_labels_last else None
                ),
                is_ordinal=architecture
                in (
                    Architecture.ORDINAL,
                    Architecture.REGULARIZED,
                    Architecture.CONTINUOUS_AUG,
                ),
            )


@app.command()
def dataset_info(matpath: str) -> None:
    """
    Print a dataset summary: total FOV count, image dimensions, and per-class
    FOV counts, global index ranges, and patient IDs.
    """
    from cli.dataset_info import run

    run(matpath)


@app.command()
def infer(
    artifact_dir: str,
    matpath: str,
    fov_index: int = typer.Option(
        ..., "--fov", "-F", help="FOV index to run inference on"
    ),
) -> None:
    """
    Run MIL inference on a single FOV and save a 5-panel attention heatmap
    visualisation to {artifact_dir}/inference_fov{N}.png.

    Requires model weights saved during training (lampe-cli train ... -s).
    The hyperparams.json in the parent of artifact_dir is read automatically.
    """
    from cli.infer import run

    run(artifact_dir, matpath, fov_index)


@app.command()
def view(
    matpath: str,
    fov_index: int = typer.Option(..., "--fov", "-F", help="FOV index to visualise"),
    output_dir: str = typer.Option(".", "--output-dir", "-o", help="Directory to save the PNG"),
) -> None:
    """
    Render a 5-panel FOV figure for a single data point without running inference.

    Saves view_fov{N}.png to output_dir (default: current directory).
    The attention heatmap panel is blank; the title shows only the true label.
    """
    from cli.view import run

    run(matpath, fov_index, output_dir)


@app.command()
def healthcheck():
    """
    Simple health check command to verify that the CLI is working.
    """
    print("LAMPE CLI is up and running!")
