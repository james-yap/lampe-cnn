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

## Directory structure

- `docs/`: Documentation files, including original LAMPE research paper and dataset/methodology descriptions.
- `playground/`: Jupyter notebooks and scripts for data exploration, preprocessing, and model development.
- `lampe_dataset/`: Place the dataset here. Multimodal stimulated Raman scattering microscopy images (not tracked by Git) used to train the CNN.