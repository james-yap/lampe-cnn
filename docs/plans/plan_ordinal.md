# Plan: Ordinal Architecture (`ordinal.py`)

## Motivation

The four tissue classes — Healthy, LGC, HGC, IDC — are **ordinal**: they represent strictly increasing biological severity. Standard multi-class `CrossEntropyLoss` treats all classes as independent and equally spaced, discarding this structure. A confusion between Healthy and IDC (two extremes) is penalised identically to a confusion between LGC and HGC (adjacent grades) — which is clinically wrong.

The **K-1 ordinal encoding rule** (also called the cumulative binary decomposition) decomposes the K-class problem into K-1 independent binary questions:

| Threshold task | Question | Healthy | LGC | HGC | IDC |
|---|---|:---:|:---:|:---:|:---:|
| Task 0 | Is severity ≥ LGC? | 0 | 1 | 1 | 1 |
| Task 1 | Is severity ≥ HGC? | 0 | 0 | 1 | 1 |
| Task 2 | Is severity ≥ IDC? | 0 | 0 | 0 | 1 |

So the ordinal label vectors are:

| Class | Label vector |
|---|---|
| Healthy (0) | `[0, 0, 0]` |
| LGC (1) | `[1, 0, 0]` |
| HGC (2) | `[1, 1, 0]` |
| IDC (3) | `[1, 1, 1]` |

This encoding has a key structural property: every valid label vector is a prefix of ones followed by zeros. The model never needs to decide how far apart the grades are numerically — only which thresholds are exceeded. This also means that prediction errors are naturally penalised proportionally to how wrong they are (confusing Healthy with IDC fires 3 binary tasks wrong; confusing LGC with HGC fires only 1).

This architecture is built on top of `class_balanced.py`, inheriting all improvements: sliding window, z-score normalization, augmentation, and `WeightedRandomSampler`.

---

## Architecture Design

### New file: `src/architectures/ordinal.py`

Inherits all logic from `class_balanced.py`. Key additions:

- `OrdinalDatapoint` type alias: `(patch, ordinal_label_tensor, class_label_int, patient_id)` — the 4-tuple is necessary because the training loop needs ordinal labels for loss, while metrics need the original integer class labels
- `encode_ordinal(label, num_classes)` — module-level helper converting integer class to label vector
- `decode_ordinal(logits)` — module-level helper converting K-1 logits back to integer class index
- `__getitem__` returns the 4-tuple; augmentation still only on training split
- Model head outputs K-1 = 3 logits (not 4)
- Loss: `BCEWithLogitsLoss` (binary cross-entropy per threshold task, averaged)
- ROC curves: one curve per binary threshold task (3 curves) instead of per-class

### CLI addition

- New `Architecture.ORDINAL = "ordinal"` enum value
- Training loop unpacks 4-tuple from DataLoader
- Loss uses `BCEWithLogitsLoss`; metrics decoded via `decode_ordinal`
- `WeightedRandomSampler` reused (identical to `class_balanced`)

---

## Phase 1 — Helpers and Type Alias

### 1.1 Type alias

The DataLoader now yields 4-tuples. Define a new type alias at module level to keep the code self-documenting:

```python
# (patch tensor, ordinal label vector, original class index, patient id)
OrdinalDatapoint = tuple[torch.Tensor, torch.Tensor, int, str]
```

### 1.2 `encode_ordinal` helper

```python
def encode_ordinal(label: int, num_classes: int = 4) -> torch.Tensor:
    """
    Encode an integer class label into a K-1 ordinal binary vector.

    Example (num_classes=4):
        0 (Healthy) -> [0., 0., 0.]
        1 (LGC)     -> [1., 0., 0.]
        2 (HGC)     -> [1., 1., 0.]
        3 (IDC)     -> [1., 1., 1.]
    """
    return torch.tensor(
        [1.0 if label > k else 0.0 for k in range(num_classes - 1)],
        dtype=torch.float,
    )
```

### 1.3 `decode_ordinal` helper

At inference, the model outputs K-1 raw logits. Convert to a class prediction:

```python
def decode_ordinal(logits: torch.Tensor) -> torch.Tensor:
    """
    Decode K-1 ordinal logits to integer class predictions.

    Each binary task independently answers "is severity >= threshold k+1?".
    The predicted class is the sum of tasks predicted as positive (>= 0.5 after sigmoid).

    Args:
        logits: Tensor of shape (batch, K-1) — raw (pre-sigmoid) model outputs.

    Returns:
        Tensor of shape (batch,) with integer class predictions in [0, K-1].
    """
    probs = torch.sigmoid(logits)          # (batch, K-1)
    return (probs > 0.5).sum(dim=1).long() # (batch,)  values in {0, 1, 2, 3}
```

---

## Phase 2 — `OrdinalDataset`

### 2.1 Class signature

```python
class OrdinalDataset(Dataset[OrdinalDatapoint]):
    """
    Sliding window dataset with per-channel z-score normalization,
    geometric augmentation (train split only), per-patch WeightedRandomSampler
    weights, and ordinal K-1 label encoding.
    """
    mat_reader: MatReader
    window_size: int
    stride: int
    num_patches_per_fov: int
    top_left_coords: list[tuple[int, int]]
    mean: torch.Tensor
    std: torch.Tensor
    train: bool
    sample_weights: torch.Tensor
```

### 2.2 Constructor

Identical to `ClassBalancedDataset.__init__` — copy verbatim. No changes needed at construction time; the ordinal encoding is applied lazily in `__getitem__`.

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
    ...  # identical to ClassBalancedDataset
```

### 2.3 `__getitem__` — 4-tuple return

The only structural difference from `ClassBalancedDataset.__getitem__` is encoding the label and returning a 4-tuple:

```python
def __getitem__(self, idx: int) -> OrdinalDatapoint:
    fov_idx = self.eff_fov_indices[idx // self.num_patches_per_fov]
    patch_idx = idx % self.num_patches_per_fov
    y, x = self.top_left_coords[patch_idx]

    patch = self.mat_reader.images[
        fov_idx, :, y : y + self.window_size, x : x + self.window_size
    ]
    patch_tensor = torch.from_numpy(patch).float()         # (C, 224, 224)
    class_label = int(self.mat_reader.class_labels[fov_idx])
    patient_id = str(self.mat_reader.patient_ids[fov_idx])

    # Standardize
    patch_tensor = (patch_tensor - self.mean[:, None, None]) / self.std[:, None, None]

    # Geometric augmentation — train split only
    if self.train:
        if torch.rand(1).item() > 0.5:
            patch_tensor = torch.flip(patch_tensor, dims=[2])  # horizontal flip
        if torch.rand(1).item() > 0.5:
            patch_tensor = torch.flip(patch_tensor, dims=[1])  # vertical flip
        k = int(torch.randint(0, 4, (1,)).item())
        if k > 0:
            patch_tensor = torch.rot90(patch_tensor, k=k, dims=[1, 2])

    # Ordinal K-1 encoding
    ordinal_label = encode_ordinal(class_label)  # (K-1,) = (3,)

    return patch_tensor, ordinal_label, class_label, patient_id
```

---

## Phase 3 — Model Factory

### 3.1 Output head change

The model now outputs **K-1 = 3 logits** (one per binary threshold task), not 4:

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
    num_ordinal_outputs = num_classes - 1  # K-1 binary threshold tasks
    model.fc = nn.Sequential(
        nn.Dropout(0.5), nn.Linear(num_ftrs, num_ordinal_outputs)
    )
    return model
```

### 3.2 Why K-1 outputs and BCEWithLogitsLoss?

Each of the K-1 outputs is an independent logistic regression over a severity threshold. `BCEWithLogitsLoss` applies a sigmoid to each logit and computes binary cross-entropy, then averages over all K-1 tasks. This means:
- A perfectly correct prediction for a sample contributes 0 loss
- A one-step-off prediction for one threshold contributes a small loss (one of three tasks wrong)
- A maximally wrong prediction (e.g., Healthy predicted as IDC) contributes a large loss (all three tasks wrong)

This is the central advantage over `CrossEntropyLoss`: the loss magnitude is **proportional to the ordinal distance of the error**.

---

## Phase 4 — CLI Integration (`typer_entrypoint.py`)

### 4.1 Add to Architecture enum

```python
class Architecture(str, Enum):
    SLIDING_WINDOW = "sliding_window"
    STANDARDIZED = "standardized"
    CLASS_BALANCED = "class_balanced"
    ORDINAL = "ordinal"               # NEW
```

### 4.2 Lazy import

```python
from architectures import sliding_window, standardized, class_balanced, ordinal
```

### 4.3 Dataset and sampler instantiation in the fold loop

```python
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
        mean_override=ord_train.mean,
        std_override=ord_train.std,
    )
    train_subset = ord_train
    model = ordinal.get_model(num_classes=NUM_CLASSES).to(device)
    sampler = WeightedRandomSampler(
        weights=ord_train.sample_weights.tolist(),
        num_samples=len(ord_train),
        replacement=True,
    )
    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler)
```

### 4.4 Training loop changes

The DataLoader for `ORDINAL` yields 4-tuples. The inner loop needs to unpack the extra element and use `BCEWithLogitsLoss`:

**Loss instantiation** (per fold, conditional):
```python
if architecture == Architecture.ORDINAL:
    criterion = torch.nn.BCEWithLogitsLoss()
else:
    criterion = torch.nn.CrossEntropyLoss()
```

**Training inner loop** (patching the unpacking for the ORDINAL case):
```python
for batch in train_loader:
    if architecture == Architecture.ORDINAL:
        patches, ordinal_labels, _class_labels, _patient_ids = batch
        patches = patches.to(device)
        targets = ordinal_labels.to(device)          # (batch, 3) float
    else:
        patches, labels, _patient_ids = batch
        patches, targets = patches.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(patches)
    loss = criterion(outputs, targets)
    ...
```

**Validation inner loop** — similarly unpack the 4-tuple, and decode ordinal logits back to class indices for metric accumulation:
```python
for batch in val_loader:
    if architecture == Architecture.ORDINAL:
        patches, ordinal_labels, class_labels, _patient_ids = batch
        patches = patches.to(device)
        targets = ordinal_labels.to(device)
        outputs = model(patches)
        loss = criterion(outputs, targets)
        # Decode to class indices for metrics
        decoded_preds = ordinal.decode_ordinal(outputs.cpu())   # (batch,) int
        all_preds.append(decoded_preds.unsqueeze(1).float())    # store as logit-like for compat
        all_labels.append(class_labels)
    else:
        patches, labels, _patient_ids = batch
        ...
```

> **Note on `all_preds` shape**: the existing metric code calls `all_preds_cat.argmax(dim=1)`. For ordinal, we instead pass the already-decoded class predictions. The cleanest approach is to make the `all_preds` list store the decoded class predictions directly (as a 1-column tensor) and special-case the `argmax` call. This is captured in phase 4 tasks below.

### 4.5 ROC curve change for ordinal architecture

The existing ROC code iterates over `range(NUM_CLASSES)` and uses `all_probs[:, i]` (a softmax probability per class). For ordinal:
- We have 3 sigmoid outputs (one per threshold task), not 4 softmax values
- The most meaningful ROC to report is one per binary threshold task: "Is severity ≥ LGC?", "Is severity ≥ HGC?", "Is severity ≥ IDC?"
- Or we can compute per-binary-class ROC by binarizing the decoded class predictions against each class (one-vs-rest on the decoded integer predictions)

The cleanest approach for comparability with the other architectures is to keep the 4-class one-vs-rest ROC scheme but use the binary thresholded probabilities to reconstruct per-class probabilities:

| Class | P(predict == class) approximation |
|---|---|
| Healthy (0) | `1 - σ(logit_0)` |
| LGC (1) | `σ(logit_0) - σ(logit_1)` |
| HGC (2) | `σ(logit_1) - σ(logit_2)` |
| IDC (3) | `σ(logit_2)` |

These reconstruct the probability mass function from the cumulative probabilities. This preserves the same plotting interface as the other architectures.

---

## Phase 5 — Documentation

- Update `research.md` section 6.4 to describe `OrdinalDataset`
- Update Summary Diagram
- Add `ordinal` row to Trial Matrix in `README.md`
- Update CLI usage table in `README.md`

---

## Design Decisions Summary

| Decision | Choice | Rationale |
|---|---|---|
| Base class | `class_balanced.py` | Inherits normalization, augmentation, and oversampling; one variable changed per experiment |
| Return type | 4-tuple `(patch, ordinal_label, class_label, patient_id)` | Training needs ordinal labels for BCEWithLogitsLoss; metrics need integer class labels |
| Loss | `BCEWithLogitsLoss` | K-1 independent binary tasks; loss magnitude scales with ordinal error distance |
| Number of model outputs | K-1 = 3 | One logit per threshold task, not one per class |
| Decoding | `sum(sigmoid(logits) > 0.5)` | Sums how many thresholds are exceeded; always produces a value in {0,1,2,3} |
| ROC curve reconstruction | Cumulative → class probs via differencing σ values | Maintains same plotting interface as other architectures for fair visual comparison |
| WeightedRandomSampler | Reused unchanged | Class imbalance still present; ordinal loss doesn't inherently address sampling bias |
| `unfreeze_layer3` flag | Present, default False | Consistent with `class_balanced`; enables future fine-tuning experiments |

---

## Todo List

### Phase 1 — Helpers and Type Alias

- [x] **1.1** Create `src/architectures/ordinal.py` by copying `class_balanced.py`
- [x] **1.2** Rename class from `ClassBalancedDataset` to `OrdinalDataset` and update module docstring
- [x] **1.3** Define `OrdinalDatapoint = tuple[torch.Tensor, torch.Tensor, int, str]` type alias at module level (replace the old `Datapoint` alias)
- [x] **1.4** Implement `encode_ordinal(label: int, num_classes: int = 4) -> torch.Tensor` helper at module level
- [x] **1.5** Implement `decode_ordinal(logits: torch.Tensor) -> torch.Tensor` helper at module level

### Phase 2 — `OrdinalDataset`

- [x] **2.1** Update `Dataset[Datapoint]` → `Dataset[OrdinalDatapoint]` in class declaration
- [x] **2.2** Update `__len__` and `__init__` (constructor body is identical to `ClassBalancedDataset` — no changes needed to constructor logic)
- [x] **2.3** Update `__getitem__` return type annotation to `OrdinalDatapoint`
- [x] **2.4** Add `ordinal_label = encode_ordinal(class_label)` call in `__getitem__` after augmentation
- [x] **2.5** Change the return statement to the 4-tuple: `return patch_tensor, ordinal_label, class_label, patient_id`
- [x] **2.6** Update `__init__` docstring to reference ordinal encoding

### Phase 3 — Model Factory

- [x] **3.1** Add `num_ordinal_outputs = num_classes - 1` inside `get_model()`
- [x] **3.2** Change the `nn.Linear` output size from `num_classes` to `num_ordinal_outputs`
- [x] **3.3** Update `get_model()` docstring to explain K-1 output semantics and why `BCEWithLogitsLoss` is used

### Phase 4 — CLI Integration

- [x] **4.1** Add `ORDINAL = "ordinal"` to the `Architecture` enum
- [x] **4.2** Add `ordinal` to the lazy import line inside `train()`
- [x] **4.3** Add `elif architecture == Architecture.ORDINAL:` branch in the fold loop (dataset, model, sampler, train_loader) — identical sampler logic to `CLASS_BALANCED`
- [x] **4.4** Conditionally select `BCEWithLogitsLoss` vs `CrossEntropyLoss` based on architecture, just before the epoch loop
- [x] **4.5** Refactor the training inner loop to handle both 3-tuple (existing) and 4-tuple (ordinal) DataLoader output
- [x] **4.6** Refactor the validation inner loop to unpack 4-tuple and decode ordinal logits back to class indices via `ordinal.decode_ordinal`
- [x] **4.7** Store per-class probability estimates for ordinal (reconstructed from cumulative sigmoid) in `all_probs` for ROC curve compatibility
- [x] **4.8** Guard the `all_preds_cat.argmax(dim=1)` metric call — for ordinal, predictions are already class indices (no argmax needed)

### Phase 5 — Documentation

- [x] **5.1** Add section 6.4 to `research.md` describing `OrdinalDataset`, the K-1 encoding, loss, and decoding
- [x] **5.2** Update the Summary Diagram in `research.md`
- [x] **5.3** Add `ordinal` row to Trial Matrix in `README.md`
- [x] **5.4** Update the Architecture entry in the `README.md` CLI usage table

### Phase 6 — Validation (post-implementation)

- [x] **6.1** Run `uv run pyright src/` — expect 0 errors
- [ ] **6.2** Run `lampe-cli train ordinal "lampe_dataset/Full images/"` and verify class distribution is printed and K-1 outputs are confirmed in the first epoch log
- [ ] **6.3** Confirm that loss decreases monotonically in early epochs (sanity check that BCEWithLogitsLoss is receiving the right tensor shapes)
- [ ] **6.4** Confirm no data leakage assertion fires
- [ ] **6.5** Confirm artifact folder name contains `ordinal`
- [ ] **6.6** Compare confusion matrices between `class_balanced` and `ordinal` — errors should cluster near the diagonal
