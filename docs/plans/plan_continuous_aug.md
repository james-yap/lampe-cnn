# Plan: Continuous Augmentation Architecture

## Motivation

From `artifacts/03_22-16_07-regularized/observation.txt`:

> Current implementation: Translational augmentation is fixed.
> Create a new architecture based on "regularized" that randomizes translation
> augmentation. Ensure continuous geometric augmentations on both rotation and
> translation instead of fixed angles/offsets. Inject tiny amount of random
> Gaussian noise to create infinite number of unique tensors without destroying
> the underlying spectral signature of the SRS and SHG microscopy data.

### Root Problem: Discrete Augmentation Creates Memorisable Duplicates

The current `OrdinalDataset.__getitem__` augmentation pipeline (used by `regularized.py`)
has three operations, all discrete:

```python
# Current — only 3 axes of variation total
if torch.rand(1).item() > 0.5:
    patch_tensor = torch.flip(patch_tensor, dims=[2])   # hflip: 2 choices
if torch.rand(1).item() > 0.5:
    patch_tensor = torch.flip(patch_tensor, dims=[1])   # vflip: 2 choices
k = int(torch.randint(0, 4, (1,)).item())
if k > 0:
    patch_tensor = torch.rot90(patch_tensor, k=k, dims=[1, 2])  # 4 choices
```

Combined, this gives at most 2 × 2 × 4 = **16 unique augmented versions** of any
given patch. With `WeightedRandomSampler` oversampling the Healthy class by ~9.5×,
the same 100 Healthy FOVs produce ~950 training samples per epoch. With only 16
distinct variants, every unique augmentation is likely seen multiple times per epoch
— a direct invitation for the model to memorise Healthy patches rather than
generalise them.

### Pigeonhole Analysis

| Class | FOV count (example fold) | Oversampled count (~950 target) | Unique augmented variants | Expected repeats per epoch |
|---|---|---|---|---|
| Healthy | ~100 FOVs × 25 patches = 2,500 patches | ~950 | 16 per patch | ~6× per unique variant |
| IDC | ~275 FOVs × 25 patches = 6,875 patches | ~950 | 16 per patch | <1 (subsampled) |

The Healthy class is oversampled *and* has a tiny augmentation space. The fix is to
make the augmentation space effectively infinite, so no two training samples from the
same patch are identical, even across hundreds of epochs.

### Why This Won't Destroy Spectral Integrity

SRS and SHG microscopy images carry **quantitative spectral intensity values** per
pixel. The three augmentation operations proposed below preserve this:

- **Continuous rotation (affine)**: rotates pixel coordinates, uses bilinear
  interpolation to fill gaps. Pixel values themselves are unchanged; only their
  positions shift. The spectral meaning of any given pixel value is invariant to
  spatial rotation.
- **Random crop instead of fixed stride offset**: the selected 224×224 window position
  within the FOV changes each call, effectively varying translation continuously.
  No pixel values are modified.
- **Additive Gaussian noise (σ ≈ 0.01–0.02, applied post-normalisation)**: the signal
  SNR for SRS images at typical acquisition settings is >30 dB. A noise level of σ=0.02
  on normalised data corresponds to ~0.02 standardised units — far below the
  signal-level differences between tissue classes. The biological structure of the
  spectrum is preserved; only the exact floating-point tensor value changes.

---

## Architecture Overview

New file: `src/architectures/continuous_aug.py`

- Inherits everything from `regularized.py` (which re-exports from `ordinal.py`):
  ordinal K-1 encoding, sample weights, z-score normalisation, phased unfreezing,
  weight decay, `ReduceLROnPlateau`.
- Replaces the fixed-stride pre-computed grid with **random crop** at `__getitem__`
  time (continuous translation).
- Replaces discrete `rot90` with **continuous rotation** via `torchvision.transforms.functional.rotate`.
- Keeps flips (they remain valid and cheap).
- Adds **per-sample additive Gaussian noise** (train split only).
- `get_model()` is identical to `regularized.get_model()` — re-exported unchanged.

### Key Dataset Differences

| Property | `OrdinalDataset` (base) | `ContinuousAugDataset` (new) |
|---|---|---|
| Translation | Fixed grid: `top_left_coords` pre-computed at init | Random crop: `(y, x)` sampled uniformly per call |
| Rotation | Discrete: `rot90` (0°, 90°, 180°, 270°) | Continuous: uniform `U(0°, 360°)` with bilinear interpolation |
| Noise | None | Additive Gaussian, σ=0.02, post-normalisation, train only |
| Augmentation space | 16 variants per patch | Effectively infinite |
| `num_patches_per_fov` | 25 (factor=5) | Still 25 — used for sample weight calculation only; actual patches are random |

---

## Implementation Details

### 1. Constructor Change: Remove Pre-computed Grid

In the current `OrdinalDataset`, `top_left_coords` is computed at init as a fixed
`factor × factor` grid. In `ContinuousAugDataset`, the same list is kept for:
- maintaining `num_patches_per_fov = factor²` (used for sample weight sizes)
- the **validation split**, which uses fixed coordinates for reproducibility

The train split ignores this grid entirely in `__getitem__`.

```python
# In __init__ — keep the grid for sample weight sizing and val reproducibility
self.top_left_coords: list[tuple[int, int]] = []
for y in range(0, height - self.window_size + 1, self.stride):
    for x in range(0, width - self.window_size + 1, self.stride):
        self.top_left_coords.append((y, x))
self.num_patches_per_fov = len(self.top_left_coords)

# Store max valid origin for random crop bounds
self._max_y = height - self.window_size   # inclusive upper bound for random y
self._max_x = width - self.window_size    # inclusive upper bound for random x
```

### 2. `__getitem__`: Continuous Translation

```python
def __getitem__(self, idx: int) -> OrdinalDatapoint:
    fov_idx = self.eff_fov_indices[idx // self.num_patches_per_fov]
    class_label = int(self.mat_reader.class_labels[fov_idx])
    patient_id = str(self.mat_reader.patient_ids[fov_idx])

    if self.train:
        # Continuous translation: sample a random top-left corner
        y = int(torch.randint(0, self._max_y + 1, (1,)).item())
        x = int(torch.randint(0, self._max_x + 1, (1,)).item())
    else:
        # Reproducible fixed grid for validation
        patch_idx = idx % self.num_patches_per_fov
        y, x = self.top_left_coords[patch_idx]

    patch = self.mat_reader.images[
        fov_idx, :, y : y + self.window_size, x : x + self.window_size
    ]
    patch_tensor = torch.from_numpy(patch).float()  # (C, 224, 224)

    # Z-score normalisation
    patch_tensor = (patch_tensor - self.mean[:, None, None]) / self.std[:, None, None]

    if self.train:
        # Flips (discrete — cheap and still valid)
        if torch.rand(1).item() > 0.5:
            patch_tensor = torch.flip(patch_tensor, dims=[2])  # hflip
        if torch.rand(1).item() > 0.5:
            patch_tensor = torch.flip(patch_tensor, dims=[1])  # vflip

        # Continuous rotation: uniform angle in [0, 360)
        angle = torch.rand(1).item() * 360.0
        patch_tensor = TF.rotate(
            patch_tensor,
            angle=angle,
            interpolation=TF.InterpolationMode.BILINEAR,
            fill=0.0,   # fill corners with 0 (= mean-normalised background)
        )

        # Additive Gaussian noise: σ=0.02 on normalised data
        # This is small relative to class-discriminative signal but ensures
        # every tensor is unique, preventing exact memorisation.
        patch_tensor = patch_tensor + torch.randn_like(patch_tensor) * 0.02

    ordinal_label = encode_ordinal(class_label)
    return patch_tensor, ordinal_label, class_label, patient_id
```

### 3. Import Addition

```python
import torchvision.transforms.functional as TF
```

### 4. `get_model()`: Re-export Unchanged

`continuous_aug.py` re-exports `get_model` from `regularized.py` unchanged, exactly as
`regularized.py` does from `ordinal.py`:

```python
from architectures.regularized import (  # noqa: F401
    OrdinalDatapoint,
    CLASS_NAMES,
    encode_ordinal,
    decode_ordinal,
    get_model,
)
```

The `ContinuousAugDataset` exposes the same public interface as `OrdinalDataset`:
`sample_weights`, `mean`, `std`, `num_patches_per_fov`.

### 5. `OptimizerEngine.for_architecture()` Update

Add `"continuous_aug"` to the flag mapping with the same flags as `"regularized"`:

```python
# In shared/optimizer_engine.py, for_architecture() method:
use_ordinal_loss = architecture in ("ordinal", "regularized", "continuous_aug")
use_weight_decay = architecture in ("regularized", "continuous_aug")
use_scheduler    = architecture in ("regularized", "continuous_aug")
use_phased_unfreezing = architecture in ("regularized", "continuous_aug")
```

**Updated flag table:**

| Architecture | ordinal loss | weight decay | scheduler | phased unfreezing |
|---|---|---|---|---|
| `sliding_window` | ✗ | ✗ | ✗ | ✗ |
| `standardized` | ✗ | ✗ | ✗ | ✗ |
| `class_balanced` | ✗ | ✗ | ✗ | ✗ |
| `ordinal` | ✓ | ✗ | ✗ | ✗ |
| `regularized` | ✓ | ✓ | ✓ | ✓ |
| `continuous_aug` | ✓ | ✓ | ✓ | ✓ |

### 6. CLI Updates (`typer_entrypoint.py`)

```python
# Architecture enum
CONTINUOUS_AUG = "continuous_aug"

# Lazy import
from architectures import ..., continuous_aug

# Fold branch (mirrors REGULARIZED exactly, replacing module name)
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
    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler)

# is_ordinal check — add CONTINUOUS_AUG alongside ORDINAL and REGULARIZED
is_ordinal=architecture in (
    Architecture.ORDINAL, Architecture.REGULARIZED, Architecture.CONTINUOUS_AUG
)
```

---

## Noise Magnitude Justification

The SRS microscopy images are normalised to z-scores using the training split mean/std.
A σ=0.02 noise injection corresponds to:

$$\text{SNR impact} = 20 \log_{10}\left(\frac{1}{0.02}\right) \approx 34 \text{ dB}$$

Neither `docs/paper/paper.pdf` nor `docs/paper/supplement.pdf` provide a numerical
acquisition SNR figure. The paper describes the chosen Raman bands (1450 cm⁻¹ and
1668 cm⁻¹) as having "prominent peak heights providing a strong signal-to-noise ratio
(SNR) in the SRS images relative to other Raman bands in the fingerprint region" — a
qualitative description only. However, two facts from the paper do support that the
underlying images have low raw noise:

1. **10-frame averaging**: each image is constructed by averaging 10 successively
   imaged frames at 20 µs pixel dwell time. Frame averaging reduces shot noise by
   $1/\sqrt{10} \approx 0.32\times$, substantially improving raw SNR before any
   normalisation.
2. **Off-resonance background subtraction**: a background image is subtracted from
   each on-resonance image, which further reduces non-vibrationally-resonant noise
   contributions (cross-phase modulation, two-photon absorption, thermal lensing).

After z-score normalisation with training-set mean/std, typical per-channel values
span a dynamic range of several standardised units between tissue classes. A σ=0.02
injection is therefore approximately 1–2% of the inter-class signal range — a
conservative, empirically motivated starting point. No quantitative claim about the
instrument noise floor should be inferred from this number.

The parameter must be treated as tunable. If val loss increases or classification
metrics degrade relative to the `regularized` baseline, reduce to σ=0.01 or remove
the noise term entirely.

---

## Changes Required

### New Files

- `src/architectures/continuous_aug.py`

### Modified Files

- `src/shared/optimizer_engine.py` — add `"continuous_aug"` to flag expressions in `for_architecture()`
- `src/cli/typer_entrypoint.py` — add `CONTINUOUS_AUG` enum value, lazy import, fold branch, `is_ordinal` extension

### `docs/research.md`

- Section 4 (package tree): add `continuous_aug.py`
- Section 6: add Section 6.6 describing `ContinuousAugDataset`
- Section 5 (`OptimizerEngine` flag table): add `continuous_aug` row
- Section 12 (Planned/Implemented): move `continuous_aug` from Planned to Implemented after this is done

---

## Todo List

### `ContinuousAugDataset`

- [x] **CA.1** Create `src/architectures/continuous_aug.py`
- [x] **CA.2** Add module docstring explaining all three augmentation changes vs. `regularized`
- [x] **CA.3** Import `torchvision.transforms.functional as TF` and all other required symbols
- [x] **CA.4** Re-export `OrdinalDatapoint`, `CLASS_NAMES`, `encode_ordinal`, `decode_ordinal`, `get_model` from `architectures.regularized`
- [x] **CA.5** Implement `ContinuousAugDataset(Dataset[OrdinalDatapoint])`:
  - [x] **CA.5a** `__init__`: copy constructor from `OrdinalDataset`; add `_max_y` / `_max_x` attributes
  - [x] **CA.5b** `__len__`: identical to `OrdinalDataset`
  - [x] **CA.5c** `__getitem__` train path: random crop origin, hflip, vflip, continuous rotation `U(0°, 360°)`, Gaussian noise σ=0.02
  - [x] **CA.5d** `__getitem__` val path: fixed `top_left_coords` grid (identical to `OrdinalDataset`)

### `OptimizerEngine`

- [x] **CA.6** Update `for_architecture()` in `src/shared/optimizer_engine.py`:
  - Add `"continuous_aug"` to each of the four `in (...)` membership expressions
  - Update the docstring flag table

### CLI

- [x] **CA.7** Add `CONTINUOUS_AUG = "continuous_aug"` to `Architecture` enum
- [x] **CA.8** Add `continuous_aug` to lazy import list
- [x] **CA.9** Add fold branch for `CONTINUOUS_AUG` (mirrors `REGULARIZED`)
- [x] **CA.10** Add `Architecture.CONTINUOUS_AUG` to all three `architecture in (...)` guards (train batch, val batch, `is_ordinal` reporter call)

### Validation

- [x] **CA.11** Run `uv run pyright src/` — 0 errors, 0 warnings confirmed
- [ ] **CA.12** Smoke run `lampe-cli train continuous_aug <matpath> -e 5` — verify:
  - Dataset construction prints class distribution
  - Training runs without error
  - Phase 2 message appears at epoch `head_only_epochs`
  - `results.png` generates with train + val confusion matrices

### Documentation

- [x] **CA.13** Update `docs/research.md`: package tree, Section 6.6, OptimizerEngine flag table, Section 12
