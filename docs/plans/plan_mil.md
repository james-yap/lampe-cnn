# Plan: Multiple Instance Learning (MIL) Architecture

## Interpretation of Regularized Results

### What Improved vs. Ordinal

The regularization worked at the level it was targeting:

| Symptom | Ordinal run | Regularized run |
|---|---|---|
| Val loss at epoch 0 | Flat/rising immediately | **Descends before plateauing** ✓ |
| Train loss final | 0.09–0.11 | 0.26–0.29 (less severe) ✓ |
| Early stopping | ~5 epochs (trivial) | 10–13 epochs (training actually progresses) ✓ |
| Class 0 recall | 0.00–0.01 | 0.01–0.05 (marginal gain) ≈ |
| AUC Class 0 | ~0.50 | **0.80–0.82** ✓ |

The scheduler and phased unfreezing successfully broke the instant overfitting loop. The model now learns meaningful features — AUC of 0.80–0.82 for Class 0 (Healthy) is strong evidence that discriminative signal is being captured. The backbone features are being used.

### What Is Still Broken

Despite the improved val loss curve, the classification performance on minority classes remains poor:

| Class | Fold 2 F1 | Fold 6 F1 | Notes |
|---|---|---|---|
| 0 – Healthy | 0.09 | 0.02 | Near-zero recall; the model almost never predicts Healthy |
| 1 – LGC | 0.26 | 0.08 | Marginal; most samples predicted as HGC/IDC |
| 2 – HGC | 0.47 | 0.60 | Over-represented; model defaults here |
| 3 – IDC | 0.62 | 0.56 | Best class; benefits from clear IDC features |

**Macro F1 of 0.31–0.36** — above chance but far from clinically useful.

---

## Root Cause Analysis

### 1. High AUC + Low Recall = Threshold Miscalibration, Not Feature Failure

Class 0 AUC is 0.80–0.82, meaning the model is **correctly ranking** Healthy patches above cancerous ones in ~80% of comparisons. It has learned the features. The problem is the decision boundary is miscalibrated: the model never crosses the threshold into predicting class 0.

The ordinal K-1 encoding amplifies this. For a patch to be predicted as Healthy (class 0), **all three** binary threshold tasks must output < 0.5:

```
predict Healthy  iff  σ(logit_0) < 0.5  AND  σ(logit_1) < 0.5  AND  σ(logit_2) < 0.5
```

This is a **conjunction of three under-predictions** — much harder to satisfy than a single argmax. Any threshold that drifts slightly toward 0.5 due to uncertainty produces a class 1+ prediction. The ordinal encoding is inherently biased against the lowest class.

### 2. Patch-Level Labels are Noisy (Weak Supervision)

Every patch in a Healthy FOV is labeled Healthy — but some of those 224×224 windows may contain tissue regions that genuinely resemble early-grade cancer. The model sees these confusing patches during training and learns to hedge. At inference it defaults to the majority class (HGC) as a safe bet.

This is the **multiple instance learning (MIL) problem**: the FOV-level label is ground truth, but the patch-level labels are inherited and therefore noisy. The model is trained to solve the wrong problem (patch classification) when the real signal is at the FOV level.

### 3. Remaining Train/Val Gap

Train loss 0.26–0.29 vs val loss 0.47–0.51 — a factor of ~1.8× ratio. The model is still memorising some patch-specific features that don't transfer across patients. This is consistent with the small number of unique patients (groups) per fold.

---

## Architecture Direction: FOV-Level Multiple Instance Learning

### Why MIL

Multiple instance learning treats each **FOV as a bag** and each **patch as an instance**. The bag label (Healthy/LGC/HGC/IDC) is ground truth; instance labels are unknown and irrelevant. The model:

1. Extracts a feature vector from each of the 25 patches independently using the ResNet18 backbone
2. Aggregates the 25 feature vectors into a single FOV-level descriptor
3. Classifies the FOV descriptor directly

This resolves all three root causes above:

| Problem | How MIL fixes it |
|---|---|
| Threshold miscalibration | We switch back to standard `CrossEntropyLoss` on 4-class logits — no ordinal conjunction bias |
| Noisy patch labels | FOV-level classification is ground truth; patch confusion is absorbed by the aggregation |
| Overfitting | Effective dataset size drops from ~15,000 patches → ~600 FOVs per fold; far less memorisation opportunity |
| Fold stability | Variance is reduced because predictions are made per-FOV, not per-patch |

### Aggregation Strategy: Attention MIL

Plain mean-pooling works but attention-weighted aggregation (Ilse et al. 2018) is stronger and interpretable:

```
z_k  = backbone(patch_k)                         # (512,) per patch
a_k  = softmax( W_att · tanh(V_att · z_k) )      # (1,) scalar attention weight
z_bag = Σ_k  a_k * z_k                           # (512,) weighted sum
y_hat = classifier(Dropout(z_bag))               # (4,) logits
```

The attention weights `a_k` can be visualised as a heatmap over the FOV, showing which patches drove the diagnosis. This is medically useful.

### Model Architecture

```
Input:  1 FOV  →  25 patches  (25, 3, 224, 224)

Stage 1 (shared across patches):
    ResNet18 backbone (frozen except layer4)
    Global average pool → (25, 512)

Stage 2 (MIL aggregation):
    Linear(512, 128) + tanh  →  (25, 128)        # attention V
    Linear(128, 1)            →  (25, 1)          # attention W
    Softmax over 25 patches   →  (25, 1)          # normalised attention
    Weighted sum              →  (512,)           # bag descriptor

Stage 3 (classification head):
    Dropout(0.5)
    Linear(512, 4)
    → CrossEntropyLoss
```

The entire model is differentiable end-to-end: gradients flow from the classification loss back through the attention weights and into the backbone in phase 2.

### Dataset Change

A new dataset class is needed that returns **one FOV at a time** (a bag of patches), instead of one patch at a time:

```python
MILDatapoint = tuple[Tensor, int, str]
# (bag_of_patches: (N_patches, 3, 224, 224), class_label: int, patient_id: str)
```

- No ordinal encoding needed — CrossEntropyLoss on 4-class logits
- Augmentation still applied per-patch (each patch independently augmented at train time)
- Z-score normalisation still per-channel using train-fold statistics
- `WeightedRandomSampler` still needed — sample at the FOV level (one weight per FOV)

### Loss and Optimiser

- `CrossEntropyLoss` (4-class) — simpler and unbiased
- Adam with `weight_decay=1e-4`
- `ReduceLROnPlateau` — same as regularized
- Phased unfreezing: freeze backbone for `head_only_epochs` epochs (head + attention module train first), then unfreeze layer4

One important change: because training is now per-FOV, the effective batch size should be 1–4 FOVs, each containing 25 patches. This means the batch shape is `(B, 25, 3, 224, 224)` with B small.

### Expected Outcomes

| Metric | Regularized | MIL Expected |
|---|---|---|
| Class 0 recall | 0.01–0.05 | 0.20–0.40 |
| Macro F1 | 0.31–0.36 | 0.45–0.55 |
| Train/val gap | ~1.8× | ~1.2× |
| Interpretability | None | Attention heatmap per FOV |
| Fold stability | Moderate variance | Lower variance (FOV-level signal) |

---

## Secondary Direction: Ordinal + Focal Loss with Label Smoothing

If MIL is too large a change to implement first, a smaller patch-level improvement is:

1. **Replace BCEWithLogitsLoss with focal BCE** (`alpha=0.25, gamma=2.0`): down-weights easy examples, forcing the model to focus on hard class 0/1 patches instead of ignoring them once loss is acceptable
2. **Ordinal label smoothing**: instead of hard `[0., 0., 0.]` for Healthy, use `[0.05, 0.05, 0.05]` — reduces the penalty for near-miss threshold crossings and softens the conjunction bias
3. **Increase `head_only_epochs` to 5 and `patience` to 10**: give phase 1 more time to stabilise the head before layer4 is introduced

This direction is low implementation cost but unlikely to solve the root weak-supervision problem.

---

## Recommended Implementation Order

**Implement MIL** — it is the architecturally correct solution for the bag-of-patches problem and is directly motivated by the data structure (FOV-level ground truth, patch-level features).

---

## Todo List

### MIL Dataset

- [ ] **M.1** Create `src/architectures/mil.py`
- [ ] **M.2** Define `MILDatapoint = tuple[Tensor, int, str]` — `(N_patches, C, H, W)` bag tensor, class label, patient ID
- [ ] **M.3** Implement `MILDataset(Dataset[MILDatapoint])`:
  - Constructor: same signature as `OrdinalDataset` (mat_reader, eff_fov_indices, factor, train, mean_override, std_override)
  - `__len__` returns number of FOVs (not patches)
  - `__getitem__(fov_idx)` extracts all `num_patches_per_fov` patches, applies normalisation + augmentation per-patch, returns bag tensor `(N, 3, 224, 224)` + label + patient_id
  - Per-FOV sample weights for `WeightedRandomSampler` (same inverse-frequency logic, one weight per FOV)
  - Class distribution report on construction

### MIL Model

- [ ] **M.4** Implement `AttentionMIL(nn.Module)`:
  - Constructor takes `num_classes=4`, `freeze_all=False`
  - `backbone`: ResNet18 with head removed (up to avgpool → 512-dim)
  - `attention_V`: `Linear(512, 128)` + `Tanh`
  - `attention_W`: `Linear(128, 1)`
  - `classifier`: `Dropout(0.5)` + `Linear(512, num_classes)`
  - `forward(bag: Tensor) -> Tensor`: `bag` is `(N_patches, C, H, W)`; returns `(num_classes,)` logits
  - Phased unfreezing: when `freeze_all=True`, layer4 stays frozen; CLI adds it via `add_param_group`
- [ ] **M.5** Add `def get_model(num_classes=4, freeze_all=False) -> nn.Module` factory in `mil.py`

### CLI Updates

- [ ] **M.6** Add `MIL = "mil"` to `Architecture` enum in `typer_entrypoint.py`
- [ ] **M.7** Add `mil` to lazy imports
- [ ] **M.8** Add `MIL` fold branch:
  - `MILDataset` for train + val
  - `WeightedRandomSampler` at FOV level
  - `DataLoader` with `batch_size=4` (4 FOVs per step, each a bag of 25 patches) — may need `collate_fn` to stack variable-length bags
  - `get_model(freeze_all=(head_only_epochs > 0))`
- [ ] **M.9** Criterion: `CrossEntropyLoss` for MIL (no ordinal encoding)
- [ ] **M.10** Batch unpacking: `bags, int_class_labels, _patient_ids = batch` (3-tuple)
- [ ] **M.11** Phase transition block: same `add_param_group` pattern as REGULARIZED but on layer4 of the MIL backbone (`model.backbone.layer4`)
- [ ] **M.12** Prediction/metrics block: `argmax(outputs)` + `softmax` probabilities (same as non-ordinal path)

### Validation

- [ ] **M.13** Run `uv run pyright src/` — expect 0 errors
- [ ] **M.14** Smoke run: `lampe-cli train mil <matpath> -e 5 -b 4` — verify:
  - Class distribution prints once per FOV split (not per patch)
  - "Phase 2" message appears at correct epoch
  - Results plots generate without error
