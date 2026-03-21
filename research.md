# LAMPE CNN — Deep Research Report

## 1. Project Identity

| Field | Value |
|---|---|
| Course | COMP 4107 — Group 5 |
| Package name | `lampe_cnn` |
| Collaborator | LAMPE Lab, Carleton University |
| Goal | Label-free histological identification of intraductal carcinoma of the prostate (IDC) using deep learning on stimulated Raman scattering (SRS) microscopy images |
| Python | ≥ 3.13 |
| Package manager | `uv` (Astral) |
| Entry point | `lampe-cli` (Typer) |

---

## 2. Scientific Domain

The project performs **4-class histological classification of prostate tissue** from non-invasively acquired microscopy images — a "label-free" approach because it does not require histological staining.

### Classes (ordinal, increasing cancer severity)

| Index | Label | Meaning |
|---|---|---|
| 0 | Healthy | Benign tissue |
| 1 | LGC | Low-Grade Cancer |
| 2 | HGC | High-Grade Cancer |
| 3 | IDC | Intraductal Carcinoma |

The ordinal nature of the classes is noted in the README as a possible future direction (ordinal regression), but is currently treated as standard 4-class classification.

### Imaging Modalities

Each sample is a **multimodal microscopy image** composed of three co-registered channels:

| Channel key | Physical meaning |
|---|---|
| `{Class}_1450_bgsub` | SRS at 1450 cm⁻¹ (background-subtracted) — highlights lipids and proteins |
| `{Class}_1668_bgsub` | SRS at 1668 cm⁻¹ (background-subtracted) — highlights proteins/amide |
| `{Class}_SHG` | Second Harmonic Generation — highlights collagen/fibrillar structure |

These three channels are stacked as a 3-channel image (analogous to RGB) and fed into a ResNet18 architecture.

---

## 3. Dataset

### Dataset Variants

The `lampe_dataset/` directory contains several versions of the data at different processing stages:

| Subdirectory | Description |
|---|---|
| `3x3 all images/` | Pre-extracted 3×3 subimage patches, all included (noisy/bad images present) |
| `3x3 bad SHG removed/` | Same 3×3 patches but with FOVs that have bad/empty SHG channels removed. Includes `_cut_` versions. |
| `Full images/` | Full-resolution FOV images with `*_names.txt` metadata files. **Primary dataset used in training.** |
| `Image data/` | Another 3×3 patch variant (no names files) |
| `Texture Statistics Data/` | Pre-computed first- and second-order texture statistics (MATLAB `.mat`). Not used by any current code. |

### File Format

Each class has a corresponding `.mat` file (MATLAB) loaded via `scipy.io.loadmat`:
- `Healthy_bulk_data.mat`, `LGC_bulk_data.mat`, `HGC_bulk_data.mat`, `IDC_bulk_data.mat`

Within a `.mat` file, each modality is stored under a key like `Healthy_1450_bgsub`, with shape `(height, width, n_samples)` (MATLAB column-major ordering). After loading, `MatReader` transposes this to `(n_samples, n_modalities, height, width)` for PyTorch conventions.

### Naming Convention (`*_names.txt`)

Each line in a names file represents one FOV (Field of View / sample):

```
1 B1 IDC 2
```

Format: `{slide_number} {position_on_slide} {class} {fov_number}`

Patient IDs are constructed as `{slide_number}_{position_on_slide}` (e.g., `"1_B1"`). This allows identifying all FOVs belonging to the same physical patient core.

### Important Dataset Quirks

- **Cross-class patients**: Some patient cores appear in multiple classes (e.g., the same `1_B1` core has both LGC and HGC images) because a single biopsy core can contain tissue regions of different severity.
- **Class imbalance**
- **Bad SHG FOVs**: Some FOVs have empty or corrupted SHG channels. The `empty_fov_filter_example.py` playground script was built specifically to visualize these, comparing the "all images" vs. "bad SHG removed" variants.

---

## 4. Package Architecture (`src/`)

The source code is installed as an editable package (`uv pip install -e .`), rooted at `src/`. It has three sub-packages:

```
src/
├── __init__.py
├── architectures/
│   ├── sliding_window.py    # Full-image sliding window CNN
│   └── standardized.py     # Sliding window CNN + per-channel normalization
├── cli/
│   └── typer_entrypoint.py  # lampe-cli entry point
└── shared/
    ├── early_stopping.py    # Patience-based early stopping
    └── mat_reader.py        # MATLAB .mat file loader
```

All three sub-packages carry `py.typed` marker files (PEP 561), and type annotations are enforced via Pyright (`pyrightconfig.json`).

---

## 5. Shared Utilities

### `MatReader` (`shared/mat_reader.py`)

The central data-loading class. On construction it:

1. Validates all required `.mat` and `.txt` files for all 4 classes exist.
2. For each class, loads the `.mat` file and iterates over the 3 modalities (`1450_bgsub`, `1668_bgsub`, `SHG`).
3. Stacks the 3 modality arrays along a new axis → `(width, height, n_samples, n_modalities)`, then transposes → `(n_samples, n_modalities, height, width)`.
4. Parses the corresponding `*_names.txt` file to extract patient IDs.
5. Concatenates all classes together.

After construction, exposes three arrays as attributes:

| Attribute | Shape | Description |
|---|---|---|
| `images` | `(N, 3, H, W)` | All FOV images, all classes |
| `class_labels` | `(N,)` | Integer class index per sample |
| `patient_ids` | `(N,)` | String patient identifier per sample |

Helper methods: `get_dims()`, `get_num_fovs()`, `get_num_channels()`, `get_height_width()`.

### `EarlyStopping` (`shared/early_stopping.py`)

Simple stateful early stopping. Tracks `best_loss`. A new loss must improve by at least `delta=0.001` to reset the counter. After `patience` epochs without sufficient improvement, sets `early_stop = True`.

---

## 6. Architecture Implementations

### 6.1 `SlidingWindowDataset` (`architectures/sliding_window.py`)

**Concept**: Instead of resizing each full FOV to 224×224 (which would discard spatial resolution), this dataset extracts multiple overlapping 224×224 patches from each full-resolution image via a sliding window, preserving fine-grained spatial detail.

**Parameters**:
- `factor` (default: 5): number of patches per spatial dimension → 5×5 = **25 patches per FOV**
- `window_size`: fixed at 224 (ResNet's native input resolution)
- `stride`: computed as `(image_dim - 224) // (factor - 1)`, giving evenly-spaced overlapping patches

**Patch indexing**: All `(y, x)` top-left coordinates are pre-computed and stored in `top_left_coords`. In `__getitem__`, a flat index is decomposed as `fov_idx = eff_fov_indices[idx // num_patches_per_fov]` and `patch_idx = idx % num_patches_per_fov`.

**`eff_fov_indices`**: The dataset only surfaces FOVs whose global indices are in this list. This is the mechanism for k-fold splits — the same `MatReader` instance is shared; only the active index set differs between train and val.

**No normalization**: Patches are converted to `float32` and returned raw.

**Model** (`get_model()`):
- Base: `ResNet18` with `ResNet18_Weights.DEFAULT` (ImageNet pretrained)
- Entire network frozen; only `layer4` (the last residual block, two conv layers) and the new head are trainable
- Custom head: `Dropout(0.5) → Linear(512, 4)`
- Rationale: Overlapping patches from the same FOV are highly correlated, so heavy dropout prevents the model from memorizing local artifacts.

### 6.2 `StandardizedDataset` (`architectures/standardized.py`)

Identical to `SlidingWindowDataset` in structure, but adds **per-channel mean/std normalization** (z-score standardization).

**Normalization computation** (on train split only):
- Iterates over all FOVs in `eff_fov_indices`
- Accumulates per-channel pixel sums and squared sums across the full image (not just patches)
- Computes `mean = sum / num_pixels` and `std = sqrt(sum_sq / num_pixels - mean²)`
- Near-zero std values (< 1e-8) are replaced with 1.0 to avoid division-by-zero

**Data leakage prevention**: The validation set is initialized with `mean_override=train_subset.mean` and `std_override=train_subset.std`, ensuring it is normalized using only statistics derived from training data. This is correctly implemented and is a key methodological improvement over `SlidingWindowDataset`.

**In `__getitem__`**: Patch is normalized as `(patch - mean[:, None, None]) / std[:, None, None]`, where the `None` expansions broadcast the per-channel scalars across the H×W dimensions.

**Model** (`get_model()`): Identical architecture to `sliding_window.get_model()`.

### 6.3 `ClassBalancedDataset` (`architectures/class_balanced.py`)

A third architecture built directly on top of `StandardizedDataset`, adding two complementary mechanisms to address class imbalance.

**On-the-fly class imbalance detection**: At construction time, the dataset computes and prints the class distribution of the active FOV split (by count and percentage), alerting to imbalance before training begins.

**Per-patch sample weights**: Computed using inverse FOV-level class frequency. Every patch belonging to a FOV of class _c_ gets weight `1 / num_fovs_in_class_c`. The `sample_weights` tensor is exposed for use with PyTorch's `WeightedRandomSampler` in the DataLoader.

**Geometric augmentation** (training split only): Random horizontal flip, vertical flip, and 90°/180°/270° rotation are applied after normalization. These transforms are physically valid for SRS microscopy because tissue sections have no canonical orientation and the channel values carry quantitative Raman spectral information that must not be distorted by other transforms (e.g., colour jitter).

**Data leakage prevention**: Identical to `StandardizedDataset` — `mean_override`/`std_override` are passed to the validation split from the training split's computed statistics.

**CLI integration**: A `WeightedRandomSampler` is used for the training DataLoader in place of `shuffle=True`, so each epoch sees an approximately balanced class distribution regardless of the raw dataset proportions. `num_samples=len(train_subset)` keeps epoch length consistent with the other architectures for fair loss curve comparison.

**Model** (`get_model()`): Same ResNet18 as `standardized`, with an optional `unfreeze_layer3` flag for follow-up experiments. First run uses `layer4`-only fine-tuning to isolate the effect of the class balancing changes.

---

## 7. CLI (`cli/typer_entrypoint.py`)

Entry point: `lampe-cli` (registered in `pyproject.toml` as `cli.typer_entrypoint:app`).

### Commands

#### `train <architecture> <matpath> [options]`

Full training pipeline with cross-validation.

| Option | Short | Default | Description |
|---|---|---|---|
| `--num-epochs` | `-e` | 20 | Max epochs per fold |
| `--n-folds` | `-f` | 6 | Number of CV folds |
| `--batch-size` | `-b` | 32 | DataLoader batch size |
| `--lr` | `-l` | 1e-4 | Adam learning rate |
| `--patience` | `-p` | 5 | Early stopping patience |
| `--save-weights` | `-s` | False | Save model state dict |

**Architecture enum**: `sliding_window`, `standardized`, or `class_balanced` (planned).

#### `healthcheck`

Prints a status message. No-ops for verifying CLI installation.

### Training Loop Detail

1. **Artifact directory** created at `artifacts/{MM_DD-HH_MM}-{architecture}/` with `hyperparams.json`.
2. A stub `SlidingWindowDataset` is always created (even for `standardized`) with one FOV to log `num_patches_per_fov` into hyperparams.
3. **Device detection**: CUDA → MPS (Apple Silicon) → CPU.
4. `StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)` splits indices, stratified by `class_labels`, grouped by `patient_ids`.
5. Per fold:
   - Dataset and model instantiated fresh.
   - Optimizer: `Adam(lr=lr)`, loss: `CrossEntropyLoss`.
   - Per epoch: standard forward/backward/step training, then validation with `torch.no_grad()`.
   - **Runtime data leakage assertion**: during validation, each patient ID is checked against `seen_in_training`. An assertion failure immediately halts training with a clear error message.
   - **Stratification debug**: tracks `{patient_id → label}` mappings and warns (in verbose mode) on inconsistencies.
   - Early stopping checked after each epoch.
6. Per fold artifacts saved to `artifacts/.../fold-{n}/`:
   - `results.png`: 2×2 grid — loss curves, confusion matrix, per-class ROC/AUC, classification report text.
   - `model_weights.pth` (only if `--save-weights`).

### Lazy Import Strategy

All heavy imports (`torch`, `sklearn`, `matplotlib`, etc.) are inside the `train()` function body. This ensures that `lampe-cli --help` responds instantly without loading the full ML stack. Ruff linting rule `PLC0415` ("imports outside top-level") is suppressed in `pyproject.toml` to allow this pattern.

---

## 8. Standalone Baseline (`base-ResNet/`)

Early-development standalone scripts, **not integrated with the main package**:

| File | Role |
|---|---|
| `model.py` | `ResNet18()` function — pretrained, fine-tunes layer4+fc, no dropout |
| `process_data.py` | `ProstateDataset` loading 3×3 patches via PIL, `TransformedSubset` for split-specific transforms, hardcoded Windows path |
| `train.py` | Simple train/val/test loop, best-accuracy checkpoint, no k-fold CV |

Differences from the main package:
- Uses PIL instead of raw numpy/torch for image handling
- Uses `*_onres` (on-resonance) keys instead of `*_bgsub` (background-subtracted) keys
- Uses `Resize(224,224)` + `RandomHorizontalFlip/VerticalFlip/Rotation(20)` + `Normalize([0.5]*3, [0.5]*3)` via torchvision transforms
- No patient-level group splitting (susceptible to data leakage)
- No k-fold cross-validation, single random 80/10/10 split

These scripts serve as a reference implementation and pre-date the structured package.

---

## 9. Playground Scripts

| Script | Purpose |
|---|---|
| `scipy_matlab_example.py` | Load and visualize one FOV across all 3 modalities using matplotlib `jet` colormap. Demonstrates the basic `scipy.io.loadmat` workflow. |
| `empty_fov_filter_example.py` | Find all FOVs present in `3x3 all images` but absent from `3x3 bad SHG removed`. Saves each missing FOV as a PNG with all modalities side-by-side. Asserts that the identified count matches the known difference. |

---

## 10. Experimental Runs (Artifacts)

Two completed training runs are stored in `artifacts/`:

| Run | Date | Architecture | Notes |
|---|---|---|---|
| `03_18-10_52-sliding_window` | March 18, 10:52 | `sliding_window` | No normalization baseline |
| `03_19-16_57-standardized` | March 19, 16:57 | `standardized` | Per-channel z-score normalization |

Both runs share identical hyperparameters:

```json
{
  "num_epochs": 20,
  "n_folds": 6,
  "batch_size": 32,
  "learning_rate": 0.0001,
  "early_stopping_patience": 5,
  "sliding_factor": 5,
  "num_patches_per_fov": 25
}
```

Each run contains 6 fold subdirectories, each with a `results.png` showing loss curves, confusion matrix, ROC curves, and classification report. No model weights are saved (default `--save-weights=False`).

The 25 patches per FOV imply the full images are large enough that with factor=5, there is meaningful overlap between adjacent patches (stride < 224).

---

## 11. Design Decisions & Notable Specifics

### Correctness & Safety

- **Data leakage guard (runtime)**: The assertion `pid not in seen_in_training` during validation is a strong correctness guarantee — training is immediately aborted if any patient crosses the train/val boundary.
- **Normalization leakage guard**: `StandardizedDataset` accepts `mean_override`/`std_override` to allow validation normalization to use only training statistics.
- **Reproducibility**: `StratifiedGroupKFold` uses `random_state=42`, ensuring the same fold splits across runs with the same dataset.
- **`torch.clamp(variance, min=0.0)`**: Prevents negative variance due to floating-point rounding errors before taking the square root.

### Architecture Rationale (Documented in Code)

- ResNet18 chosen over deeper variants (ResNet50, etc.) because sliding window patches from the same FOV are highly correlated — a larger model would overfit more severely.
- Freezing the backbone except `layer4` leverages pre-trained low-level edge/texture detectors from ImageNet while allowing higher-level feature adaptation.
- `Dropout(0.5)` in the classification head further regularizes against the patch-correlation problem.

### Serialization of Enum in JSON

`Architecture` inherits from both `str` and `Enum` (`class Architecture(str, Enum)`), so `json.dump` serializes it as its string value (e.g., `"sliding_window"`) without a custom encoder. This is an intentional design choice.

---

## 12. Missing / Incomplete / Future Work

### Not Implemented (per professor feedback in `docs/proposal_feedback.txt`)

| Requirement | Status |
|---|---|
| Automatic feature selection | Not implemented |
| Multi-modal fusion (explicit) | Partial — channels are stacked but no explicit fusion module |
| GradCAM visualization | Not implemented |

### Not Implemented (per README notes)

| Item | Status |
|---|---|
| `inference` CLI command | Mentioned in README, no code exists |
| Learning rate scheduling (StepLR, CosineAnnealingLR) | Not implemented |
| Fixed held-out test set | Not implemented |
| Ensemble binary classifiers | Not implemented |
| Ordinal regression loss | Not implemented |
| Regularization (weight decay, etc.) | Not implemented |

### In Progress / Implemented

- `class_balanced.py`: Addresses class imbalance via geometric augmentation and `WeightedRandomSampler`. See `plan.md` for full specification.

### Incomplete Modules

- `base-ResNet/process_data.py`: Ends mid-script (split logic incomplete).
- `Texture Statistics Data/`: Pre-computed features exist on disk but no code reads or uses them.

---

## 13. Dependency Inventory

```toml
torch >= 2.10.0
torchvision >= 0.25.0
numpy >= 2.4.3
scipy >= 1.17.1
scikit-learn >= 1.8.0
matplotlib >= 3.10.8
typer >= 0.24.1
```

All versions are strictly pinned for reproducibility via `uv`.

---

## 14. Summary Diagram

```
lampe-cli train <arch> <matpath>
        │
        ├── MatReader(matpath)
        │       └── loads Healthy/LGC/HGC/IDC .mat files
        │           stacks 3 modalities per FOV
        │           extracts patient IDs from *_names.txt
        │           → images(N,3,H,W), class_labels(N,), patient_ids(N,)
        │
        ├── StratifiedGroupKFold(n_splits=6, groups=patient_ids)
        │       └── ensures no patient spans train+val boundary
        │
        └── for each fold:
                ├── {Sliding|Standardized|ClassBalanced}Dataset(mat_reader, eff_fov_indices)
                │       └── 25 overlapping 224×224 patches per FOV
                │           [Standardized/ClassBalanced: z-score normalize w/ train stats]
                │           [ClassBalanced only: prints class distribution, computes sample_weights]
                │           [ClassBalanced only: hflip/vflip/rot90 augmentation (train split)]
                │
                ├── ResNet18(pretrained) → freeze all → unfreeze layer4
                │       └── head: Dropout(0.5) → Linear(512, 4)
                │
                ├── Adam(lr=1e-4) + CrossEntropyLoss + EarlyStopping(patience=5)
                │   [ClassBalanced only: WeightedRandomSampler on train_loader]
                │
                └── artifacts/fold-{n}/
                        ├── results.png  (loss curves, CM, ROC, report)
                        └── model_weights.pth  (if --save-weights)
```
