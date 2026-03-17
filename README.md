# COMP 4107 – Group 5

Label-free histological identification of intraductal carcinoma of the prostate using deep learning on multimodal stimulated Raman scattering microscopy (LAMPE Lab @ Carleton University)

## Getting Started

### Environment

This project is bootstrapped with [uv](https://docs.astral.sh/uv/guides/projects/#creating-a-new-project). Go through [this guide](https://docs.astral.sh/uv/guides/projects/#creating-a-new-project) for more information.

### Adding dependencies

Strict versioning is used for dependencies to ensure reproducibility and is tracked by `pyproject.toml`. To add a new dependency, run the following command:

```bash
uv add <package-name>
```

### Running scripts

Use `uv run` to execute scripts.

```bash
uv run playground/scipy_matlab_example.py
```

## Development Setup

The source code lives under `src/` and is structured as a Python package (`lampe_cnn`). For imports to resolve correctly — both at runtime and in your editor — the package must be installed in **editable mode**:

```bash
uv pip install -e .
```

This registers `src/` as the package root so that `shared`, `architectures`, and `cli` are importable from any script. Re-run this command whenever `pyproject.toml` changes (e.g. after adding a dependency).

For VS Code / Pylance to resolve imports correctly, ensure the interpreter is set to the project's virtual environment:

1. `Cmd+Shift+P` → **Python: Select Interpreter**
2. Select `.venv/bin/python` (shown as `('.venv': venv)`)

## CLI Usage

The `lampe-cli` tool is installed automatically as part of the editable install.

```bash
# Train a model
uv run lampe-cli train <architecture> <matpath>

# Example
uv run lampe-cli train sliding_window "lampe_dataset/Full images"

# Run inference
uv run lampe-cli inference
```

| Command | Argument | Description |
|:---|:---|:---|
| `train` | `architecture` | Architecture name (e.g. `sliding_window`) |
| `train` | `matpath` | Path to the directory containing `.mat` dataset files |
| `inference` | — | Run inference (in progress) |

## Directory Structure

```
.
├── src/
│   ├── architectures/      # Model architecture definitions (e.g. sliding window CNN)
│   ├── cli/                # Typer CLI entry point (lampe-cli)
│   └── shared/             # Shared utilities (e.g. MatReader for loading .mat files)
├── base-ResNet/            # Standalone baseline ResNet training scripts
├── playground/             # Exploratory scripts and examples
├── lampe_dataset/          # Dataset directory (not tracked by Git)
│   ├── 3x3 all images/
│   ├── 3x3 bad SHG removed/
│   ├── Full images/
│   ├── Image data/
│   └── Texture Statistics Data/
├── docs/                   # Project documentation and proposal feedback
├── pyproject.toml          # Project metadata and dependencies
└── pyrightconfig.json      # Pylance/Pyright configuration
```

- `src/architectures/`: PyTorch `Dataset` and model architecture implementations. Currently includes `SlidingWindowDataset` for patch extraction from full-size images.
- `src/cli/`: Typer-based CLI exposing `train` and `inference` commands via the `lampe-cli` entry point.
- `src/shared/`: Shared utilities. `MatReader` loads `.mat` files from a directory, groups samples by patient, and provides a unified interface for `DataLoader` consumption.
- `base-ResNet/`: Early-stage standalone scripts for a baseline ResNet model (not integrated with the main package).
- `playground/`: One-off scripts for data exploration and testing ideas before integrating them into `src/`.
- `lampe_dataset/`: Place the dataset here. Multimodal stimulated Raman scattering microscopy images (not tracked by Git) used to train the CNN.

## Trial Matrix

| Technique | Accuracy | Notes |
|:---:|:---:|:---:|
| Vanilla (use "3x3 bad SHG removed") | ? | Establish baseline |
| Native ResNet50 resolution (224 x 224) | ? | Combine with sliding window approach on full images |
| StratifiedGroupKFold | ? | Ensure balanced representation of classes and groups in training/validation splits (intracore bias) |

### Notes

- [ ] Normalization: Per-channel mean/std normalization based on training set statistics (not from ResNet)
- [ ] Learning rate scheduling: Experiment with schedulers (e.g. StepLR, CosineAnnealingLR) to improve convergence
- [ ] Geometric Augmentation
- [ ] Accuracy, recall, precision, F1 score (due to class imbalance)
- [ ] [ROC and AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)
- [ ] perhaps k-folds=5 not suitable for small dataset
- [ ] Ensemble approach: binary classifier for each class (e.g. IDC vs non-IDC, benign vs non-benign) and combine predictions
- [ ] Tune learning rate