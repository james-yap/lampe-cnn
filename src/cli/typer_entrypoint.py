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
    LINEAR_SVM = "lsvm"


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
    from shared.train import run

    run(
        architecture=architecture.value,
        matpath=matpath,
        num_epochs=num_epochs,
        n_folds=n_folds,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        save_weights=save_weights,
        weight_decay=weight_decay,
        head_only_epochs=head_only_epochs,
        num_classes=NUM_CLASSES,
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
    output_dir: str = typer.Option(
        ".", "--output-dir", "-o", help="Directory to save the PNG"
    ),
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
