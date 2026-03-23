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
│   ├── sliding_window.py      # Full-image sliding window CNN
│   ├── standardized.py        # Sliding window CNN + per-channel normalization
│   ├── class_balanced.py      # + class imbalance handling, geometric augmentation
│   ├── ordinal.py             # + K-1 ordinal label encoding
│   ├── regularized.py        # + weight decay, phased unfreezing, ReduceLROnPlateau
│   ├── continuous_aug.py     # + continuous rotation/translation, Gaussian noise
│   └── mil.py                # FOV-level attention MIL — addresses HGC collapse
├── cli/
│   └── typer_entrypoint.py    # lampe-cli entry point
└── shared/
    ├── early_stopping.py      # Patience-based early stopping
    ├── mat_reader.py          # MATLAB .mat file loader
    ├── report.py              # FoldReporter — fold evaluation visualisation
    └── optimizer_engine.py    # OptimizerEngine — architecture-aware optimizer config
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

### `FoldReporter` (`shared/report.py`)

Stateless helper that owns all per-fold evaluation, reporting, and visualisation logic. Accepts raw model output tensors and produces the standard `results.png` artifact. Hides all `matplotlib` and `sklearn.metrics` imports inside its `save()` method to preserve the lazy-import / fast CLI startup property.

Key interface:
```python
class FoldReporter:
    def __init__(self, num_classes: int, class_names: list[str]) -> None
    def save(
        self,
        fold: int,
        output_dir: str,
        train_losses: list[float],
        val_losses: list[float],
        all_preds: torch.Tensor,         # (N, num_classes) or (N, K-1)
        all_labels: torch.Tensor,        # (N,) integer class labels
        train_preds: torch.Tensor | None = None,   # last-epoch train outputs
        train_labels: torch.Tensor | None = None,  # last-epoch train labels
        is_ordinal: bool = False,
    ) -> None
```

The `is_ordinal` flag selects between two decode paths: cumulative sigmoid + ordinal decode vs. argmax + softmax. All plot rendering (2×3 mosaic: loss curve, train confusion matrix, val confusion matrix, ROC curves, classification report text) and file I/O are encapsulated inside `save()`.

### `OptimizerEngine` (`shared/optimizer_engine.py`)

Stateful object that encapsulates all optimizer, loss criterion, learning rate scheduler, and phased-unfreezing logic for one fold's training run. Constructed via a `for_architecture()` classmethod that translates an `Architecture` enum value into a set of boolean capability flags, keeping the architecture-specific conditional logic in one place.

Key interface:
```python
class OptimizerEngine:
    criterion: torch.nn.Module        # BCEWithLogitsLoss or CrossEntropyLoss
    optimizer: torch.optim.Optimizer  # Adam, configured per architecture

    @classmethod
    def for_architecture(cls, architecture, model, lr, weight_decay, head_only_epochs) -> "OptimizerEngine"
    def step_scheduler(self, val_loss: float) -> None    # no-op if no scheduler
    def maybe_transition_phase(self, epoch: int) -> bool # True on phase-2 transition epoch
```

Current architecture → flag mapping:

| Architecture | ordinal loss | weight decay | scheduler | phased unfreezing |
|---|---|---|---|---|
| `SLIDING_WINDOW` | ✗ | ✗ | ✗ | ✗ |
| `STANDARDIZED` | ✗ | ✗ | ✗ | ✗ |
| `CLASS_BALANCED` | ✗ | ✗ | ✗ | ✗ |
| `ORDINAL` | ✓ | ✗ | ✗ | ✗ |
| `REGULARIZED` | ✓ | ✓ | ✓ | ✓ |
| `CONTINUOUS_AUG` | ✓ | ✓ | ✓ | ✓ |
| `MIL` | ✗ | ✓ | ✓ | ✓ |

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

### 6.4 `OrdinalDataset` (`architectures/ordinal.py`)

Built directly on top of `class_balanced.py`, inheriting all improvements (sliding window, z-score normalization, augmentation, `WeightedRandomSampler`). The single structural change is replacing standard 4-class label encoding with **K-1 ordinal binary encoding**.

**Encoding (`encode_ordinal`)**: Each integer class label is converted to a 3-element binary float vector where position `k` answers "Is severity strictly greater than grade k?". The result is always a valid prefix of 1s followed by 0s:

| Class | Label vector |
|---|---|
| Healthy (0) | `[0, 0, 0]` |
| LGC (1) | `[1, 0, 0]` |
| HGC (2) | `[1, 1, 0]` |
| IDC (3) | `[1, 1, 1]` |

**Decoding (`decode_ordinal`)**: `sigmoid(logits) > 0.5` gives binary threshold decisions; their sum gives the integer class prediction, always in `{0, 1, 2, 3}`.

**Return type**: `__getitem__` returns a 4-tuple `(patch, ordinal_label_vector, class_label_int, patient_id)`. The training loop consumes `ordinal_label_vector` for loss; evaluation metrics use `class_label_int`.

**Loss**: `BCEWithLogitsLoss` — K-1 independent binary cross-entropies averaged together. Loss magnitude is naturally proportional to ordinal error distance: a Healthy/IDC confusion fires all 3 tasks wrong; an adjacent-grade confusion fires only 1.

**Model head**: Outputs K-1 = 3 logits (not 4). The `get_model()` factory uses `num_ordinal_outputs = num_classes - 1` for the `nn.Linear` output dimension.

**ROC curve compatibility**: Per-class probabilities are reconstructed from cumulative sigmoid differences: `P(class=k) ≈ σ(logit_{k-1}) - σ(logit_k)`, giving a valid 4-class probability mass function for the same ROC plotting interface as the other architectures.

### 6.5 `regularized.py` — Regularized Ordinal Architecture

Built on top of `ordinal.py` by re-exporting `OrdinalDataset`, `OrdinalDatapoint`, `CLASS_NAMES`, `encode_ordinal`, and `decode_ordinal` unchanged, and providing an extended `get_model()` factory that supports phased unfreezing:

```python
def get_model(num_classes=4, unfreeze_layer3=False, freeze_all=False) -> nn.Module
```

When `freeze_all=True`, layer4 stays frozen from epoch 0; the CLI (or `OptimizerEngine`) unfreezes it at the phase boundary via `optimizer.add_param_group()`.

This architecture is paired with `OptimizerEngine` flags: weight decay, `ReduceLROnPlateau` scheduler, and phased unfreezing all active.

### 6.6 `continuous_aug.py` — Continuous Augmentation Architecture

Builds on `regularized.py` by replacing the discrete fixed-stride augmentation pipeline with three augmentations that make the probability of seeing an identical training tensor twice effectively zero:

1. **Random crop (continuous translation)**: At `__getitem__` time, the train split samples a uniformly random top-left corner `(y, x)` from the full valid crop range `[0, H−224] × [0, W−224]`. The validation split retains the fixed stride grid for reproducibility.

2. **Continuous rotation** (`TF.rotate`, `U(0°, 360°)`): Replaces `rot90` (4 discrete values) with a bilinear-interpolated rotation at a uniformly random angle. Corners are padded with `0.0` (the z-score normalised background value).

3. **Additive Gaussian noise** (σ=0.02, post-normalisation, train only): A small per-sample noise sample is added after z-score normalisation. This is approximately 1–2% of the typical inter-class z-score signal range — conservative enough to preserve spectral texture while guaranteeing every training tensor is unique.

**Root cause addressed**: The previous discrete pipeline yielded at most 2 × 2 × 4 = 16 unique augmented variants per patch. With `WeightedRandomSampler` oversampling the Healthy class by ~9.5×, the same patch could be seen in identical form dozens of times per epoch, creating a direct memorisation path. `ContinuousAugDataset` closes this by making the augmentation space infinite.

**Dataset interface**: `ContinuousAugDataset` exposes the same attributes as `OrdinalDataset` (`sample_weights`, `mean`, `std`, `num_patches_per_fov`) so the CLI fold branch requires no structural changes beyond the class name.

**Re-exports**: `OrdinalDatapoint`, `CLASS_NAMES`, `encode_ordinal`, `decode_ordinal`, and `get_model` are all re-exported from `regularized.py` unchanged.

### 6.7 `mil.py` — Attention-Based Multiple Instance Learning

**Motivation**: Continuous augmentation confirmed the failure mode is architectural, not augmentation-related. The HGC collapse pattern (model predicting class 2 the vast majority of the time regardless of true label) is identical across all four folds of both `regularized` and `continuous_aug`. The root cause is weak supervision: every patch in a Healthy FOV is labeled Healthy, but some 224×224 windows contain tissue features that genuinely resemble early-grade cancer. Training a patch classifier with these noisy inherited labels teaches the model to hedge toward the majority class.

**MIL framing**: Each FOV is treated as a **bag** of patches; the bag label (pathologist-assigned ground truth) is used for training; individual patch labels are never seen by the loss function. This directly addresses the weak-supervision problem and additionally eliminates the ordinal conjunction bias by switching to a standard 4-class `CrossEntropyLoss`.

**`MILDatapoint`** type: `tuple[torch.Tensor, int, str]` — a 3-tuple of *(bag tensor `(N_patches, C, H, W)`, integer class label, patient ID)*. Unlike the ordinal architectures there is no label-vector element, so the CLI batch unpacking is a 3-tuple throughout.

**`MILDataset`**: `Dataset[MILDatapoint]` where `__len__` returns the number of FOVs (not patches), so `WeightedRandomSampler` operates at FOV granularity (one weight per FOV, not per patch). `__getitem__(idx)` extracts all `num_patches_per_fov` patches for the FOV, applies z-score normalisation using train-fold statistics, and in the train split applies per-patch continuous augmentation (random crop, `U(0°, 360°)` rotation, Gaussian noise σ=0.02) independently to each patch in the bag. The validation split uses the fixed stride grid with no augmentation. The class distribution report and `mean`/`std`/`sample_weights` interface are identical to `ContinuousAugDataset` at the FOV level.

**`AttentionMIL`** model (Ilse et al. 2018):

```
Input: bags (B, N_patches, C, H, W)

Stage 1 — shared backbone:
    ResNet18 (ImageNet pretrained), fc → Identity
    All backbone params frozen; layer4 optionally unfrozen at phase 2
    view(B*N, C, H, W) → backbone → view(B, N, 512)

Stage 2 — attention aggregation:
    attention_V: Linear(512, 128) + Tanh   → (B, N, 128)
    attention_W: Linear(128, 1)            → (B, N, 1)
    softmax(dim=1)                         → (B, N, 1)   # normalised over patches
    z_bag = (attn * feats).sum(dim=1)      → (B, 512)    # weighted bag descriptor

Stage 3 — classification head:
    Dropout(0.5) + Linear(512, num_classes) → (B, num_classes)
```

`forward(bags)` returns `(logits (B, num_classes), attn_out (B, N_patches))`. The attention weights are differentiable — gradients flow through them and into the backbone in phase 2. At inference time they directly encode the per-patch spatial importance for the 5-panel heatmap visualisation.

**`get_model(num_classes=4, freeze_all=False)`**: factory that returns an `AttentionMIL` instance. `freeze_all=True` keeps the entire backbone frozen for head-only phase 1; `OptimizerEngine.maybe_transition_phase()` detects `"layer4" in name` on `model.named_parameters()` and adds `backbone.layer4` as a new optimizer param group at `lr * 0.1` at epoch `head_only_epochs`.

**CLI batch shape**: `(B, N_patches, C, H, W)` — bags stack cleanly since every FOV produces the same `num_patches_per_fov` patches, so no custom `collate_fn` is needed. Typical values: B=4 FOVs, N=25 patches, C=3, H=W=224.

---

## 7. CLI (`cli/typer_entrypoint.py`)

Entry point: `lampe-cli` (registered in `pyproject.toml` as `cli.typer_entrypoint:app`).

### Commands

#### `train <architecture> <matpath> [options]`

Full training pipeline with cross-validation.

| Option | Short | Default | Description |
|---|---|---|---|
| `--num-epochs` | `-e` | 60 | Max epochs per fold |
| `--n-folds` | `-f` | 4 | Number of CV folds |
| `--batch-size` | `-b` | 32 | DataLoader batch size |
| `--lr` | `-l` | 1e-4 | Adam learning rate |
| `--patience` | `-p` | 10 | Early stopping patience |
| `--save-weights` | `-s` | False | Save model state dict |
| `--weight-decay` | `-w` | 1e-4 | L2 weight decay (regularized/continuous_aug/mil) |
| `--head-only-epochs` | `-H` | 50 | Head-only training epochs before layer4 unfreezes (regularized/continuous_aug/mil) |

**Architecture enum**: `sliding_window`, `standardized`, `class_balanced`, `ordinal`, `regularized`, `continuous_aug`, or `mil`.

#### `dataset-info <matpath>`

Prints a compact dataset summary with no model required:
- Total FOV count, image spatial dimensions, channel count
- Per-class table: FOV count, global index range, patient IDs (first 5 + total)

Useful for picking a valid `--fov` index before running `infer`.

#### `infer <artifact_dir> <matpath> --fov <N>`

Runs MIL inference on a single FOV and produces a 5-panel attention heatmap figure.

| Argument / Option | Description |
|---|---|
| `artifact_dir` | Fold artifact directory (e.g. `artifacts/03_22-17_05-mil/fold-2`). Must contain `model_weights.pth`. |
| `matpath` | Path to the `Full images/` directory (same as `train`). |
| `--fov` / `-F` | Required. Integer index of the FOV to run inference on. |

Prints the same dataset summary as `dataset-info`, then runs the model and saves `{artifact_dir}/inference_fov{N}.png` at 300 dpi. Raises a clear `FileNotFoundError` if weights are missing ("Run training with `-s` to save weights"). Only supports MIL artifacts; raises `ValueError` for any other architecture.

#### `healthcheck`

Prints a status message. Used for verifying CLI installation.

### Training Loop Detail

1. **Artifact directory** created at `artifacts/{MM_DD-HH_MM}-{architecture}/` with `hyperparams.json`.
2. A stub `SlidingWindowDataset` is always created (even for `standardized`) with one FOV to log `num_patches_per_fov` into hyperparams.
3. **Device detection**: CUDA → MPS (Apple Silicon) → CPU.
4. `StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)` splits indices, stratified by `class_labels`, grouped by `patient_ids`.
5. Per fold:
   - Dataset and model instantiated fresh per architecture branch.
   - `OptimizerEngine.for_architecture(...)` constructs the criterion, optimizer, and optional scheduler in one call.
   - `WeightedRandomSampler` applied to training DataLoader for `CLASS_BALANCED`, `ORDINAL`, `REGULARIZED`, `CONTINUOUS_AUG`, and `MIL` (at the FOV level for MIL).
   - Per epoch:
     - `engine.maybe_transition_phase(epoch)` fires phase-2 layer4 unfreezing at `head_only_epochs`.
     - Training inner loop has three branches: MIL (3-tuple batch, `(bags, int_class_labels, _patient_ids)`, model returns `(logits, attn)` tuple that is unpacked before the loss call); ordinal/regularized/continuous_aug (4-tuple batch, `BCEWithLogitsLoss`); all other architectures (3-tuple batch, `CrossEntropyLoss`). The val loop mirrors this structure; `_attn` is discarded.
     - Standard forward/backward/step training, then validation with `torch.no_grad()`.
     - **Runtime data leakage assertion**: during validation, each patient ID is checked against `seen_in_training`.
     - **Stratification debug**: tracks `{patient_id → label}` mappings and warns (in verbose mode) on inconsistencies.
     - Early stopping and `engine.step_scheduler(val_loss)` called after each epoch.
6. Per fold artifacts saved to `artifacts/.../fold-{n}/` via `FoldReporter.save(...)`:
   - `results.png`: 2×3 mosaic — loss curves, train confusion matrix, val confusion matrix, ROC curves, classification report text.
   - `model_weights.pth` (only if `--save-weights`).

### Inference Pipeline Detail (`infer` command)

1. Reads `hyperparams.json` from the parent directory of `artifact_dir` to recover `sliding_factor` and `architecture`. Raises if architecture is not `"mil"`.
2. Loads `MatReader` and prints the same dataset summary as `dataset-info`.
3. Validates `--fov` is in range; prints the true class and patient ID.
4. Loads `model_weights.pth` into an `AttentionMIL` model via `torch.load(..., weights_only=True)`.
5. Constructs `MILDataset(train=False)` for the single FOV (fixed stride grid, no augmentation).
6. Calls `model(bag.unsqueeze(0))` with `torch.no_grad()`, squeezes `logits (num_classes,)` and `attn (N_patches,)`.
7. **Attention heatmap construction**:
   - Reshapes `attn (N_patches,)` → `(factor, factor)` in row-major order (same order as `top_left_coords`).
   - Bilinear-upsamples to `(H, W)` via `scipy.ndimage.zoom(order=1)`.
   - Applies `matplotlib.cm.hot` colormap → `(H, W, 3)` RGB in `[0, 1]`.
   - Blends: `overlay = composite_rgb * 0.55 + heatmap_rgb * 0.45`.
8. **5-panel 1×5 figure** at 25×5 inches:
   - Panel 1: SRS1 — Lipids (channel 0, 1450 cm⁻¹), grayscale, min-max normalised
   - Panel 2: SRS2 — Proteins (channel 1, 1668 cm⁻¹), grayscale, min-max normalised
   - Panel 3: SHG — Collagen (channel 2), grayscale, min-max normalised
   - Panel 4: Composite pseudo-RGB (R=SRS1, G=SRS2, B=SHG, each channel independently min-max normalised)
   - Panel 5: Attention overlay (composite + hot-colormap heatmap at 45% opacity) with colorbar showing raw attention weight scale
9. Suptitle: `FOV N | True: {class} ({idx}) | Predicted: {class} ({idx}) | Confidence: {prob:.1%}`
10. Saves to `{artifact_dir}/inference_fov{N}.png` at 300 dpi.

### Lazy Import Strategy

All heavy imports (`torch`, `sklearn`, `matplotlib`, `scipy`, etc.) are inside the command function bodies. This ensures that `lampe-cli --help` responds instantly without loading the full ML stack. Ruff linting rule `PLC0415` ("imports outside top-level") is suppressed in `pyproject.toml` to allow this pattern.

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
| `inference` CLI command | Mentioned in README, no code exists — full spec now in `docs/plans/plan_mil.md` (`lampe-cli infer` + `lampe-cli dataset-info`) |
| Fixed held-out test set | Not implemented |
| Ensemble binary classifiers | Not implemented |
| GradCAM / attention visualization | Not implemented |
| MIL (Multiple Instance Learning) | Planned — see `docs/plans/plan_mil.md` (updated with continuous_aug results; heatmap artifact specified) |

### Implemented

- `class_balanced.py`: Addresses class imbalance via geometric augmentation and `WeightedRandomSampler`.
- `ordinal.py`: K-1 ordinal label encoding, `BCEWithLogitsLoss`, ordinal decode.
- `regularized.py`: Phased layer unfreezing, weight decay, `ReduceLROnPlateau`.
- `continuous_aug.py`: Continuous rotation/translation, Gaussian noise — breaks discrete augmentation pigeonhole.
- `shared/report.py` (`FoldReporter`): fold evaluation / visualisation with 2×3 mosaic (loss, train CM, val CM, ROC, classification report).
- `shared/optimizer_engine.py` (`OptimizerEngine`): architecture-aware optimizer, criterion, scheduler, and phased-unfreezing.
- `--weight-decay` (`-w`) and `--head-only-epochs` (`-H`) CLI options.

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
                ├── {Sliding|Standardized|ClassBalanced|Ordinal|Regularized}Dataset(mat_reader, eff_fov_indices)
                │       └── 25 overlapping 224×224 patches per FOV
                │           [Standardized+: z-score normalize w/ train stats]
                │           [ClassBalanced+: prints class distribution, computes sample_weights]
                │           [ClassBalanced+: hflip/vflip/rot90 augmentation (train split)]
                │           [Ordinal+: encode_ordinal -> (patch, ordinal_vec, class_int, pid)]
                │
                ├── get_model()  →  ResNet18(pretrained) → freeze all
                │       [Sliding/Standardized/ClassBalanced/Ordinal: unfreeze layer4 from epoch 0]
                │       [Regularized: freeze_all=True; layer4 unfrozen at epoch head_only_epochs]
                │       head: Dropout(0.5) → Linear(512, 4)  [non-ordinal]
                │             Dropout(0.5) → Linear(512, 3)  [ordinal/regularized: K-1 logits]
                │
                ├── OptimizerEngine.for_architecture(...)  ← planned shared utility
                │       └── criterion: CrossEntropyLoss [sliding/standardized/class_balanced]
                │                      BCEWithLogitsLoss [ordinal/regularized]
                │           optimizer: Adam(lr, [weight_decay=w for regularized])
                │           scheduler: ReduceLROnPlateau [regularized only]
                │           phase transition: layer4 add_param_group at epoch H [regularized only]
                │
                ├── EarlyStopping(patience)
                │   WeightedRandomSampler on train_loader [class_balanced/ordinal/regularized]
                │
                └── FoldReporter.save(...)  ← planned shared utility
                        └── artifacts/fold-{n}/
                                ├── results.png  (loss curves, CM, ROC curves, classification report)
                                └── model_weights.pth  (if --save-weights)
```
