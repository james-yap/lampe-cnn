# Plan: Multiple Instance Learning (MIL) Architecture

## Interpretation of Results Across Architectures

### Regularized → Continuous Aug: What Changed

Continuous augmentation fixed the discrete-augmentation pigeonhole problem and measurably
improved generalisation on Fold 1 (the fold that ran to epoch 60 and completed phase 2):

| Symptom | Regularized | Continuous Aug (Fold 1) |
|---|---|---|
| Val loss curve | Descends then plateaus | **Follows train closely throughout** ✓ |
| Train loss at epoch 60 | — | 0.30 (phase 2 drives sharp improvement) ✓ |
| Class 0 AUC | 0.80–0.82 | **0.84** ✓ |
| Class 3 AUC | 0.71–0.75 | **0.79** ✓ |
| Macro F1 | 0.31–0.36 | **0.34** (Fold 1), ~0.18–0.24 others |

Folds 2–4 early-stopped around epoch 25–30 (before phase 2 at epoch 50), so they never
benefit from layer4 fine-tuning. The run confirms 50 head-only epochs + 10 total is not
enough; this is a hyperparameter tuning concern, not a code change.

The close val/train curve (Fold 1) confirms continuous augmentation eliminated the
memorisable duplicate problem. The backbone is learning generalising features.

### What Is Still Broken

Despite augmentation improvements, ordinal hedging persists identically across all four folds:

| Class | Fold 1 F1 | Fold 2 F1 | Fold 3 F1 | Fold 4 F1 | Pattern |
|---|---|---|---|---|---|
| 0 – Healthy  | 0.10 | 0.00 | 0.06 | 0.15 | Near-zero recall in every fold |
| 1 – LGC      | 0.09 | 0.02 | 0.04 | 0.17 | Systematically under-predicted |
| 2 – HGC      | 0.64 | 0.56 | 0.48 | 0.54 | Model defaults here in every fold |
| 3 – IDC      | 0.53 | 0.36 | 0.14 | 0.19 | High precision but near-zero recall |

**Macro F1 of 0.18–0.34** — continuous augmentation did not solve the hedging problem
because that problem is architectural, not augmentation-related.

The HGC collapse is stable: across all four folds, the model predicts class 2 the vast
majority of the time regardless of the true label. Val confusion matrices show columns 1
and 2 absorbing almost all predictions.

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
| HGC collapse | Per-FOV aggregation sees all 25 patches simultaneously; consistent Healthy signal across the bag cannot be dismissed by hedging to HGC |

### Aggregation Strategy: Attention MIL

Plain mean-pooling works but attention-weighted aggregation (Ilse et al. 2018) is stronger and interpretable:

```
z_k  = backbone(patch_k)                         # (512,) per patch
a_k  = softmax( W_att · tanh(V_att · z_k) )      # (1,) scalar attention weight
z_bag = Σ_k  a_k * z_k                           # (512,) weighted sum
y_hat = classifier(Dropout(z_bag))               # (4,) logits
```

The attention weights `a_k` can be visualised as a heatmap over the FOV, showing which
patches drove the diagnosis. This is medically useful and is the primary output of the
`lampe-cli infer` command (see **Inference Pipeline** section below).

**Aggregate per-class heatmaps are not generated during training.** An average attention
grid collapsed across all val FOVs of a class loses the within-class spatial variability
that makes the heatmap diagnostically useful. The heatmap is only meaningful when
anchored to a specific FOV and its raw pixel data — exactly what `lampe-cli infer`
provides. During the val loop `_attn` is unpacked but discarded.

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

| Metric | Continuous Aug baseline | MIL Expected |
|---|---|---|
| Class 0 recall | 0.00–0.12 | 0.20–0.40 |
| Class 1 recall | 0.00–0.18 | 0.20–0.40 |
| Macro F1 | 0.18–0.34 | 0.45–0.55 |
| Train/val gap | Close (Fold 1) to collapsing (Fold 2) | Stable — FOV-level labels are ground truth |
| Interpretability | None | Per-FOV 5-panel visualisation at inference time |
| Fold stability | High variance across folds | Lower variance (FOV-level signal) |

---

## Secondary Direction: Ordinal + Focal Loss with Label Smoothing

> **Decision (March 2026):** This direction is not being pursued. `continuous_aug`
> confirmed the failure mode is architectural — the ordinal hedging pattern is identical
> in `regularized` and `continuous_aug` across all four folds, meaning the problem is
> weak patch-level supervision, not loss miscalibration. Focal loss or label smoothing
> cannot fix the HGC collapse because the root cause is that patch labels are noisy,
> not that the loss function weights easy examples too heavily. Preserved below for the
> record.

### Description

If MIL is too large a change to implement first, a smaller patch-level improvement is:

1. **Replace BCEWithLogitsLoss with focal BCE** (`alpha=0.25, gamma=2.0`): down-weights
   easy examples, forcing the model to focus on hard class 0/1 patches instead of
   ignoring them once loss is acceptable.
2. **Ordinal label smoothing**: instead of hard `[0., 0., 0.]` for Healthy, use
   `[0.05, 0.05, 0.05]` — reduces the penalty for near-miss threshold crossings and
   softens the conjunction bias caused by the K-1 ordinal encoding.
3. **Increase `head_only_epochs` to 5 and `patience` to 10**: give phase 1 more time
   to stabilise the head before layer4 is introduced.

This direction has low implementation cost but cannot resolve the root weak-supervision
problem: the model is being trained to solve the wrong problem (patch classification)
when the real signal is at the FOV level.

---

## Recommended Implementation Order

**Implement MIL** — it is the architecturally correct solution for the bag-of-patches problem and is directly motivated by the data structure (FOV-level ground truth, patch-level features).

---

## Implementation Notes (Updated)

`FoldReporter` (`shared/report.py`) and `OptimizerEngine` (`shared/optimizer_engine.py`)
now exist as production utilities. The CLI tasks below must use them:

- `OptimizerEngine.for_architecture("mil", ...)` must be added with flags:
  `use_ordinal_loss=False`, `use_weight_decay=True`, `use_scheduler=True`,
  `use_phased_unfreezing=True`. This is a new row in the `for_architecture()` table.
- `FoldReporter.save(...)` is called with `is_ordinal=False` for MIL (argmax + softmax path).
- No `FoldReporter` extension is needed for heatmaps — heatmap generation lives entirely
  in the `infer` CLI command.

The `MILDatapoint` type is a 3-tuple (not 4-tuple) — no ordinal label vector. All three
`architecture in (ORDINAL, REGULARIZED, CONTINUOUS_AUG)` guards in the CLI **must not**
include `MIL`.

---

## Todo List

### MIL Dataset

- [x] **M.1** Create `src/architectures/mil.py`
- [x] **M.2** Define `MILDatapoint = tuple[Tensor, int, str]` — `(N_patches, C, H, W)` bag tensor, class label, patient ID
- [x] **M.3** Implement `MILDataset(Dataset[MILDatapoint])`:
  - Constructor: same signature as `ContinuousAugDataset` (mat_reader, eff_fov_indices, factor, train, mean_override, std_override)
  - `__len__` returns number of FOVs (not patches)
  - `__getitem__(fov_idx)` extracts all `num_patches_per_fov` patches, applies z-score normalisation + continuous augmentation per-patch (same as `ContinuousAugDataset`), returns bag tensor `(N, 3, 224, 224)` + label + patient_id
  - Per-FOV sample weights for `WeightedRandomSampler` (same inverse-frequency logic, one weight per FOV not per patch)
  - Class distribution report on construction

### MIL Model

- [x] **M.4** Implement `AttentionMIL(nn.Module)`:
  - Constructor takes `num_classes=4`, `freeze_all=False`
  - `backbone`: ResNet18 with fc head removed, `avgpool` output → 512-dim feature per patch
  - `attention_V`: `Linear(512, 128)` + `Tanh`
  - `attention_W`: `Linear(128, 1)`
  - `classifier`: `Dropout(0.5)` + `Linear(512, num_classes)`
  - `forward(bag: Tensor) -> tuple[Tensor, Tensor]`: `bag` is `(N_patches, C, H, W)`; returns `((num_classes,) logits, (N_patches,) attention weights)` — attention weights are ignored during training/val but are consumed by `lampe-cli infer`
  - Phased unfreezing: when `freeze_all=True`, `backbone.layer4` stays frozen; CLI adds it via `add_param_group` on `model.backbone.layer4.parameters()`
- [x] **M.5** Add `def get_model(num_classes=4, freeze_all=False) -> nn.Module` factory in `mil.py`

### `OptimizerEngine` Update

- [x] **M.6** Add `"mil"` to `OptimizerEngine.for_architecture()` in `shared/optimizer_engine.py`:
  - `use_ordinal_loss = False`
  - `use_weight_decay = True`
  - `use_scheduler = True`
  - `use_phased_unfreezing = True`
  - Update the docstring flag table with a `mil` row

### CLI Updates

- [x] **M.8** Add `MIL = "mil"` to `Architecture` enum in `typer_entrypoint.py`
- [x] **M.9** Add `mil` to lazy imports
- [x] **M.10** Add `MIL` fold branch:
  - `MILDataset` for train + val
  - `WeightedRandomSampler` at FOV level (one weight per FOV)
  - `DataLoader` with `batch_size=4` (4 FOVs per step, each a bag of 25 patches); bags stack cleanly since all FOVs produce the same `num_patches_per_fov` patches, so no custom `collate_fn` is needed
  - `get_model(freeze_all=(head_only_epochs > 0))`
- [x] **M.11** Batch unpacking: `bags, int_class_labels, _patient_ids = batch` (3-tuple — not 4-tuple; do not add MIL to the ordinal guard)
- [x] **M.12** Phase transition block: same `engine.maybe_transition_phase(epoch)` pattern as REGULARIZED; `OptimizerEngine` handles `add_param_group` on `model.backbone.layer4`
- [x] **M.13** Val loop: unpack `logits, _attn = model(bag)` — attention weights are discarded during training/val (used only at inference time)
- [x] **M.14** `reporter.save(...)` call: `is_ordinal=False`; `train_preds` / `train_labels` collected as in other architectures

### Validation

- [x] **M.15** Run `uv run pyright src/` — expect 0 errors
- [ ] **M.16** Smoke run: `lampe-cli train mil <matpath> -e 5 -b 4` — verify:
  - Class distribution prints once per FOV split (not per patch)
  - "Phase 2" message appears at correct epoch
  - `results.png` generated without error

---

## Inference Pipeline

### Overview

After training, the `lampe-cli infer` command takes a trained MIL artifact and a single
FOV index, runs the model on that FOV without augmentation, and produces a single
`inference_fov{N}.png` containing a **5-panel visualisation** plus a classification
report.

Because heatmaps are meaningful only when anchored to specific pixel data, they are
never aggregated across FOVs during training — they are generated here, per sample,
at inference time.

### Command Signature

```
lampe-cli infer <artifact_dir> <matpath> --fov <fov_index>
```

| Argument / Option | Description |
|---|---|
| `artifact_dir` | Path to a training artifact directory (e.g. `artifacts/03_22-17_05-continuous_aug/fold-2`). Must contain `model_weights.pth`. The parent `hyperparams.json` is read for `sliding_factor` and `num_classes`. |
| `matpath` | Path to the `Full images/` directory (same argument as `train`). |
| `--fov` / `-F` | Integer index of the FOV to run inference on. |

**Prerequisite**: model weights must have been saved during training (`-s` flag). The
command will raise a clear error if `model_weights.pth` is not found.

### Dataset Info Utility

To help the user pick a valid `--fov` index, `lampe-cli infer` always prints a compact
dataset summary before running inference:

```
Dataset: lampe_dataset/Full images/
Total FOVs : 600   Image size: 512 × 512   Channels: 3
Class distribution:
  0  Healthy :  100 FOVs   indices [0 – 99]     patients: P001, P002, ...
  1  LGC     :  150 FOVs   indices [100 – 249]  patients: P003, P004, ...
  2  HGC     :  250 FOVs   indices [250 – 499]  patients: P005, P006, ...
  3  IDC     :  100 FOVs   indices [500 – 599]  patients: P007, P008, ...
Running inference on FOV 42  (class: Healthy, patient: P001)
```

This is also exposed as a standalone command:

```
lampe-cli dataset-info <matpath>
```

that prints the same summary and exits, with no model required.

### 5-Panel Visualisation Layout

The output figure is a **1 × 5** grid of image panels at full FOV resolution,
with a single-line report as the figure suptitle.

```
┌──────────┬──────────┬──────────┬──────────┬──────────────────────┐
│ SRS1     │ SRS2     │ SHG      │ Composite│ Attention Overlay    │
│ (Lipids) │(Proteins)│(Collagen)│ (RGB)    │ Composite + heatmap  │
│ grayscale│ grayscale│ grayscale│          │ ~45% opacity         │
└──────────┴──────────┴──────────┴──────────┴──────────────────────┘
FOV 42 | True: Healthy (0) | Predicted: Healthy (0) | Confidence: 94.2%
```

**Panel details:**

1. **SRS1 — Lipids** (channel 0, 1450 cm⁻¹): full FOV, single-channel, `gray`
   colormap, min-max normalised to [0, 1] across the FOV.
2. **SRS2 — Proteins** (channel 1, 1668 cm⁻¹): same as above.
3. **SHG — Collagen** (channel 2): same as above.
4. **Composite pseudo-RGB**: channels mapped R=SRS1, G=SRS2, B=SHG; each channel
   independently min-max normalised to [0, 1] then stacked into an `(H, W, 3)` array.
5. **Attention Overlay**: the composite pseudo-RGB image with the per-patch attention
   heatmap alpha-blended at 45% opacity on top.

**Attention heatmap construction for Panel 5:**
- The model returns `attn` of shape `(N_patches,)` (already softmax-normalised).
- Reshape to `(factor, factor)` using the same row-major order as `top_left_coords`.
- Upsample to `(H, W)` using `scipy.ndimage.zoom` with bilinear interpolation to match
  the full FOV resolution.
- Apply `matplotlib.cm.hot` colormap → `(H, W, 3)` RGB in [0, 1].
- Blend: `overlay = composite * 0.55 + heatmap_rgb * 0.45`.
- Axes on this panel are unlabelled; a `colorbar` is added showing the attention scale
  (min attention weight → max attention weight for this FOV).

**Suptitle** (single line below all panels):
```
FOV {fov_index}  |  True: {true_class_name} ({true_idx})  |  Predicted: {pred_class_name} ({pred_idx})  |  Confidence: {prob:.1%}
```
where `Confidence` is `softmax(logits)[pred_idx]`.

**Saved as**: `{artifact_dir}/inference_fov{fov_index}.png`, 300 dpi.

### Todo List: Inference

- [ ] **I.1** Add `dataset_info` command to `typer_entrypoint.py`:
  - `lampe-cli dataset-info <matpath>`
  - Loads `MatReader(matpath)`, prints total FOVs, image dimensions, channel count,
    and a per-class table of FOV count, index range, and patient IDs
  - No model or architecture needed

- [ ] **I.2** Add `infer` command to `typer_entrypoint.py`:
  - Signature: `artifact_dir: str`, `matpath: str`, `fov_index: int = typer.Option(..., "--fov", "-F")`
  - Loads `hyperparams.json` from `os.path.dirname(artifact_dir)` (parent of fold dir)
    to recover `sliding_factor` and `architecture`; raises clear error if architecture
    is not `"mil"`
  - Prints the dataset summary (same as `dataset-info`)
  - Loads `MatReader`, constructs `MILDataset(... train=False ...)` for the single FOV
    (so no augmentation, fixed grid)
  - Loads `model_weights.pth` from `artifact_dir`; raises `FileNotFoundError` with
    message "Run training with `-s` to save weights" if missing
  - Calls `model.eval()`, runs `logits, attn = model(bag.unsqueeze(0))`
    (adds batch dim) and squeezes results
  - Builds the 5-panel figure (see layout above)
  - Saves to `{artifact_dir}/inference_fov{fov_index}.png` at 300 dpi
  - Prints path to saved file

- [ ] **I.3** Run `uv run pyright src/` — expect 0 errors after adding both commands
