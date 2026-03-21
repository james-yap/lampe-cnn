# Plan: Class-Balanced Architecture (`class_balanced.py`)

## Motivation

The current `standardized` architecture treats all FOV patches equally during training. The dataset has known class imbalance — there are more samples for certain cancer grades and fewer for others. Training on a naively shuffled dataloader causes the model to be biased toward majority classes, under-learning the rarer pathologies.

This plan defines a new architecture that addresses class imbalance through two complementary mechanisms:

1. **Geometric augmentation** — synthetically expands the effective dataset by randomly flipping and rotating patches at training time, adding rotational/reflective invariance that is physically appropriate for SRS microscopy (no canonical tissue orientation).
2. **Weighted random oversampling** — a `WeightedRandomSampler` rebalances the class distribution seen by the optimizer in each epoch, independent of how imbalanced the raw dataset is.

The new file, `src/architectures/class_balanced.py`, is based directly on `standardized.py` so the contribution of each change is traceable in the artifact history.

---

## Architecture Design

### New file: `src/architectures/class_balanced.py`

Inherits all logic from `standardized.py` (sliding window, z-score normalization, data-leakage-safe overrides). Adds:

- `train: bool` parameter to enable augmentation only on training splits
- On-the-fly class imbalance detection and reporting in `__init__`
- Per-patch `sample_weights` tensor exposed for use with `WeightedRandomSampler`
- Augmentation block in `__getitem__` (random hflip, vflip, rot90)

### CLI addition

- New `Architecture.CLASS_BALANCED = "class_balanced"` enum value in `typer_entrypoint.py`
- Training DataLoader uses `WeightedRandomSampler` instead of `shuffle=True`
- Validation DataLoader unchanged (`shuffle=False`, no augmentation)

### Model

Same ResNet18 architecture as `standardized.get_model()`. With the increased effective diversity from both augmentation and oversampling, optionally unfreezing `layer3` in addition to `layer4` is considered (see Phase 2 notes).

---

## Phase 1 — `ClassBalancedDataset`

### 1.1 Class signature and attributes

```python
from collections import Counter

class ClassBalancedDataset(Dataset[Datapoint]):
    """
    Sliding window dataset with per-channel z-score normalization,
    geometric augmentation (train only), and per-patch sample weights
    for use with WeightedRandomSampler.
    """
    mat_reader: MatReader
    window_size: int
    stride: int
    num_patches_per_fov: int
    top_left_coords: list[tuple[int, int]]
    mean: torch.Tensor
    std: torch.Tensor
    train: bool
    sample_weights: torch.Tensor   # NEW: one weight per patch, for WeightedRandomSampler
```

### 1.2 Constructor signature

```python
def __init__(
    self,
    mat_reader: MatReader,
    eff_fov_indices: list[int],
    factor: int = 5,
    train: bool = True,
    mean_override: torch.Tensor | None = None,
    std_override: torch.Tensor | None = None,
):
```

`train=True` enables augmentation and is set to `False` for the validation split. The `mean_override`/`std_override` API is preserved identically from `StandardizedDataset`.

### 1.3 On-the-fly class imbalance detection

Placed after the stride/coordinate computation, before normalization:

```python
from collections import Counter

CLASS_NAMES = ["Healthy", "LGC", "HGC", "IDC"]

class_counts: Counter[int] = Counter(
    int(mat_reader.class_labels[i]) for i in eff_fov_indices
)
total_fovs = len(eff_fov_indices)
split_label = "train" if train else "val"
print(f"  [{split_label}] Class distribution ({total_fovs} FOVs):")
for cls_idx in range(len(CLASS_NAMES)):
    count = class_counts.get(cls_idx, 0)
    pct = 100.0 * count / total_fovs if total_fovs > 0 else 0.0
    print(f"    {CLASS_NAMES[cls_idx]:8s}: {count:4d} FOVs ({pct:.1f}%)")
```

This prints something like:

```
  [train] Class distribution (112 FOVs):
    Healthy :   42 FOVs (37.5%)
    LGC     :   18 FOVs (16.1%)
    HGC     :   28 FOVs (25.0%)
    IDC     :   24 FOVs (21.4%)
```

### 1.4 Per-patch sample weights

Computed immediately after the class distribution report, using inverse class frequency:

```python
# Build per-patch weight list: every patch from a FOV inherits that FOV's class weight.
# Weight = 1 / number_of_FOVs_in_that_class  (inverse class frequency at the FOV level)
patch_weights: list[float] = []
for fov_idx in eff_fov_indices:
    cls = int(mat_reader.class_labels[fov_idx])
    w = 1.0 / class_counts[cls]
    patch_weights.extend([w] * self.num_patches_per_fov)

self.sample_weights = torch.tensor(patch_weights, dtype=torch.float)
```

Rationale: weighting at the FOV level (rather than re-weighting 25 identical-label patches) avoids inflating the effective sample size artificially. The sampler then draws patches proportionally to produce a balanced class distribution over each epoch.

### 1.5 Normalization (unchanged from `StandardizedDataset`)

The full mean/std computation block is copied verbatim. `mean_override`/`std_override` must be passed to the validation set to prevent data leakage.

### 1.6 Geometric augmentation in `__getitem__`

Augmentation is applied after patch extraction and normalization, conditioned on `self.train`:

```python
def __getitem__(self, idx: int) -> Datapoint:
    fov_idx = self.eff_fov_indices[idx // self.num_patches_per_fov]
    patch_idx = idx % self.num_patches_per_fov
    y, x = self.top_left_coords[patch_idx]

    patch = self.mat_reader.images[
        fov_idx, :, y : y + self.window_size, x : x + self.window_size
    ]
    patch_tensor = torch.from_numpy(patch).float()         # (C, 224, 224)
    class_label = int(self.mat_reader.class_labels[fov_idx])
    patient_id = self.mat_reader.patient_ids[fov_idx]

    # Standardize (using train-split statistics)
    patch_tensor = (patch_tensor - self.mean[:, None, None]) / self.std[:, None, None]

    # Geometric augmentation — train only
    if self.train:
        if torch.rand(1).item() > 0.5:
            patch_tensor = torch.flip(patch_tensor, dims=[2])  # horizontal flip
        if torch.rand(1).item() > 0.5:
            patch_tensor = torch.flip(patch_tensor, dims=[1])  # vertical flip
        k = int(torch.randint(0, 4, (1,)).item())
        if k > 0:
            patch_tensor = torch.rot90(patch_tensor, k=k, dims=[1, 2])  # 0/90/180/270°

    return patch_tensor, class_label, patient_id
```

**Why these augmentations are valid for SRS microscopy:**
- SRS images have no inherent up/down or left/right — tissue sections are mounted arbitrarily.
- Flips and 90° rotations preserve all biologically meaningful spatial statistics.
- We avoid colour jitter, blur, or elastic deformations because the channel values carry quantitative Raman spectral information that must not be distorted.

**Why augmentation is applied after normalization:**
- Mean/std are computed on the raw pixel values. Normalizing first then augmenting is mathematically equivalent and slightly simpler.

---

## Phase 2 — Model Factory

The model factory in `class_balanced.py` can optionally unfreeze `layer3` in addition to `layer4`. With the increased effective dataset diversity from augmentation and oversampling, a slightly larger fine-tuning surface may improve feature adaptation.

```python
def get_model(num_classes: int = 4, unfreeze_layer3: bool = False) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    if unfreeze_layer3:
        for param in model.layer3.parameters():
            param.requires_grad = True

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5), nn.Linear(num_ftrs, num_classes)
    )
    return model
```

For the first run of this architecture, keep `unfreeze_layer3=False` (identical parameter count to `standardized`) to isolate the effect of the class balancing changes alone. A follow-up experiment can set `unfreeze_layer3=True`.

---

## Phase 3 — CLI Integration (`typer_entrypoint.py`)

### 3.1 Add to the Architecture enum

```python
class Architecture(str, Enum):
    SLIDING_WINDOW = "sliding_window"
    STANDARDIZED = "standardized"
    CLASS_BALANCED = "class_balanced"    # NEW
```

### 3.2 Import the new module (inside `train()`, alongside the existing lazy imports)

```python
from architectures import sliding_window, standardized, class_balanced
```

### 3.3 Dataset and sampler instantiation inside the fold loop

```python
elif architecture == Architecture.CLASS_BALANCED:
    from torch.utils.data import WeightedRandomSampler   # already available; can be top-of-function

    train_subset = class_balanced.ClassBalancedDataset(
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
        mean_override=train_subset.mean,   # prevent data leakage
        std_override=train_subset.std,
    )
    model = class_balanced.get_model(num_classes=NUM_CLASSES).to(device)
```

### 3.4 DataLoader with WeightedRandomSampler

The sampler replaces `shuffle=True` on the training loader. `shuffle` and `sampler` are mutually exclusive in PyTorch's DataLoader.

```python
if architecture == Architecture.CLASS_BALANCED:
    sampler = WeightedRandomSampler(
        weights=train_subset.sample_weights,
        num_samples=len(train_subset),
        replacement=True,
    )
    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler)
else:
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
```

`num_samples=len(train_subset)` keeps epoch length identical to the other architectures, making loss curves directly comparable.

---

## Phase 4 — `research.md` Update

- Add `class_balanced` to the Architecture Implementations section (section 6.3)
- Document: dataset class, augmentation choices, sample weights, sampler
- Add row to the Experimental Runs table when first run is executed
- Update the Summary Diagram

---

## Phase 5 — `README.md` Update

Add `class_balanced` row to the Trial Matrix:

| Technique | Accuracy | Notes |
|:---:|:---:|:---:|
| class_balanced | ? | Geometric augmentation + WeightedRandomSampler on top of standardized |

---

## Design Decisions Summary

| Decision | Choice | Rationale |
|---|---|---|
| Augmentation types | hflip, vflip, rot90(k=1,2,3) | SRS images have no canonical orientation; these preserve Raman spectral content |
| Augmentation applied to | Train split only | Val must reflect unmodified data distribution for unbiased evaluation |
| Normalization order | Normalize before augment | Equivalent result; avoids recomputing stats after augmentation |
| Weighting granularity | Per-FOV class frequency | Prevents inflating sample count; each FOV's 25 patches share one class weight |
| Sampler `num_samples` | `len(train_subset)` | Epoch length is consistent with other architectures for fair loss comparison |
| `replacement=True` | Yes (WeightedRandomSampler default) | Required for oversampling; minority patches will be seen multiple times per epoch |
| Model fine-tuning extent | layer4 only (first run) | Isolates the effect of class balancing; layer3 unfreezing reserved for follow-up |

---

## Todo List

### Phase 1 — `ClassBalancedDataset`

- [x] **1.1** Create `src/architectures/class_balanced.py` by copying `standardized.py`
- [x] **1.2** Rename class from `StandardizedDataset` to `ClassBalancedDataset` and update the module docstring
- [x] **1.3** Add `train: bool = True` parameter to `__init__` signature and class-level attribute
- [x] **1.4** Add `sample_weights: torch.Tensor` to the class-level attribute declarations
- [x] **1.5** Add `CLASS_NAMES` module-level constant (`["Healthy", "LGC", "HGC", "IDC"]`)
- [x] **1.6** Implement class distribution detection using `Counter` and print report in `__init__`
- [x] **1.7** Implement per-patch `sample_weights` computation (inverse FOV-level class frequency) in `__init__`
- [x] **1.8** Add `from collections import Counter` import at top of file
- [x] **1.9** Add geometric augmentation block in `__getitem__` after normalization (hflip, vflip, rot90), gated by `self.train`
- [x] **1.10** Update `__init__` docstring to document all new parameters and attributes

### Phase 2 — Model Factory

- [x] **2.1** Add `unfreeze_layer3: bool = False` parameter to `get_model()` in `class_balanced.py`
- [x] **2.2** Add conditional `layer3` unfreezing block inside `get_model()`
- [x] **2.3** Update `get_model()` docstring to document the new parameter and rationale

### Phase 3 — CLI Integration

- [x] **3.1** Add `CLASS_BALANCED = "class_balanced"` to the `Architecture` enum in `typer_entrypoint.py`
- [x] **3.2** Add `class_balanced` to the lazy import line inside `train()` (`from architectures import ..., class_balanced`)
- [x] **3.3** Add `WeightedRandomSampler` to the imports inside `train()`
- [x] **3.4** Add `elif architecture == Architecture.CLASS_BALANCED:` branch in the fold loop to instantiate `ClassBalancedDataset` (train and val) and `get_model()`
- [x] **3.5** Refactor the `DataLoader` instantiation so that `CLASS_BALANCED` uses `WeightedRandomSampler` (no `shuffle=True`) and other architectures use `shuffle=True` as before
- [x] **3.6** Verify the `hyperparams.json` stub logic (currently always uses `sliding_window` stub) still correctly logs `num_patches_per_fov` for the new architecture

### Phase 4 — Documentation

- [x] **4.1** Add section 6.3 to `research.md` describing `ClassBalancedDataset`
- [x] **4.2** Update the Summary Diagram in `research.md` to reflect the new architecture and sampler
- [x] **4.3** Add `class_balanced` row to the Trial Matrix in `README.md`
- [x] **4.4** Update the `Architecture` entry in the `README.md` CLI usage table

### Phase 5 — Validation

- [x] **5.1** Typecheck passes with 0 errors (pyright)
- [ ] **5.2** Run `lampe-cli train class_balanced "lampe_dataset/Full images/"` and verify class distribution is printed for each fold
- [ ] **5.3** Verify that the val loader does NOT receive augmented data (inspect output shapes and loss behaviour)
- [ ] **5.4** Confirm no data leakage assertion fires (val patient IDs never in training set)
- [ ] **5.5** Confirm artifact folder is created with correct architecture name in path
- [ ] **5.6** Compare fold `results.png` confusion matrices between `standardized` and `class_balanced` runs — minority class recall should improve
