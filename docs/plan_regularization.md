# Plan: Regularization & Learning Rate Scheduling

## Diagnosis

Both ordinal run folds show the same failure pattern:

| Symptom | Evidence |
|---|---|
| Severe overfitting | Train loss → 0.09–0.11; val loss flat or rising from epoch 0 |
| Generalization collapse | Model predicts class 2 (HGC) almost exclusively at val time |
| Minority class ignored | Class 0 (Healthy) recall = 0.00–0.01 despite WeightedRandomSampler |
| Learning rate instability | Fold 5 val loss is non-monotonic (bouncing), indicating the optimizer is taking steps too large to land in a generalizing basin |

The root cause is that the model memorizes the training patches very quickly (the ResNet18 backbone still has significant capacity even with only layer4 unfrozen), and the current training configuration has no mechanism to slow this down or penalize it.

### Why WeightedRandomSampler is not enough

The sampler ensures the optimizer *sees* balanced classes during training, but does not prevent the model from learning overly specific patch-level features that don't transfer to unseen patients. Overfitting here is a capacity/regularization problem, not a sampling problem.

---

## Two Improvement Directions

---

## Direction A: L2 Weight Decay + `ReduceLROnPlateau`

### Motivation

1. **Weight decay (L2 regularization)** adds a penalty proportional to the magnitude of model weights to the loss: $\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda \|\theta\|_2^2$. This penalises the model for memorizing training patches by driving weights toward zero, encouraging simpler, more generalisable features. In Adam, weight decay is implemented as `weight_decay` in the optimizer.

2. **`ReduceLROnPlateau`** monitors val loss and reduces the learning rate by a configurable factor whenever it stops improving for `patience` epochs. This addresses the large train/val gap caused by the learning rate being too high in the later epochs — the scheduler decays the step size precisely when the model starts overfitting.

### Changes Required

#### `src/cli/typer_entrypoint.py`

Add `weight_decay` as a CLI option:

```python
weight_decay: float = typer.Option(1e-4, "-w", help="L2 weight decay for Adam optimizer")
```

Replace the optimizer instantiation:
```python
# Before
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# After
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
```

Add a `ReduceLROnPlateau` scheduler after the optimizer:
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",       # minimize val loss
    factor=0.5,       # halve the LR on plateau
    patience=3,       # wait 3 epochs before reducing
    min_lr=1e-6,      # floor
)
```

Step the scheduler at the end of each epoch (after `early_stopping`):
```python
scheduler.step(epoch_val_loss)
```

Add `weight_decay` to `hyperparams` dict for logging.

#### Suggested default hyperparameters for this run

| Param | Current | Suggested |
|---|---|---|
| `lr` | 1e-4 | 5e-5 |
| `weight_decay` | — | 1e-4 |
| `patience` (early stopping) | 5 | 7 |
| `ReduceLROnPlateau.patience` | — | 3 |
| `ReduceLROnPlateau.factor` | — | 0.5 |

Reducing the base LR to 5e-5 slows the initial descent, giving the val loss time to follow. The scheduler then decays further if val loss stagnates.

---

## Direction B: Phased Layer Unfreezing

### Motivation

Currently layer4 is unfrozen from epoch 0. The layer4 parameters have 2.1M values, and combined with the classification head they provide enough capacity to memorise the training patches within 2–3 epochs. A **phased unfreezing** strategy starts with the head frozen-backbone training, then progressively unlocks deeper layers. This is sometimes called *discriminative fine-tuning*.

**Phase 1** (epochs 0–N₁): Freeze the entire backbone, train only the classification head (`fc`). Because only the head (~2K parameters) is free, the model cannot overfit the backbone features — it is forced to find a good linear combination of ImageNet features.

**Phase 2** (epoch N₁ onward): Unfreeze layer4. The backbone features are now well-calibrated to the domain, so fine-tuning has a stable starting point.

### Changes Required

#### `src/cli/typer_entrypoint.py`

Add a `head_only_epochs` CLI option:
```python
head_only_epochs: int = typer.Option(3, "-H", help="Epochs to train head-only before unfreezing layer4")
```

Inside the epoch loop, add a phase transition at epoch `head_only_epochs`:
```python
if epoch == head_only_epochs:
    print(f"  [Phase 2] Unfreezing layer4 at epoch {epoch + 1}")
    for param in model.layer4.parameters():
        param.requires_grad = True
    # Reset optimizer with a lower LR so unfrozen layers start with small steps
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr * 0.1,
        weight_decay=weight_decay,
    )
```

#### `get_model()` change (all architectures)

Return the model with **all layers frozen including layer4** when phased unfreezing is intended. The CLI controls when layer4 becomes trainable:

```python
def get_model(num_classes: int = 4, unfreeze_layer3: bool = False, freeze_all: bool = False) -> nn.Module:
    ...
    if not freeze_all:
        for param in model.layer4.parameters():
            param.requires_grad = True
    ...
```

When `head_only_epochs > 0`, pass `freeze_all=True` and let the CLI unfreeze layer4 at the phase boundary.

#### Suggested default hyperparameters for this run

| Param | Current | Suggested |
|---|---|---|
| `lr` | 1e-4 | 1e-4 (phase 1), 1e-5 (phase 2) |
| `head_only_epochs` | — | 3 |
| `patience` | 5 | 8 |

---

## Comparison of Directions

| Property | Direction A (Weight Decay + Scheduler) | Direction B (Phased Unfreezing) |
|---|---|---|
| Implementation complexity | Low — 5 lines changed | Medium — epoch loop restructured |
| Targets | Penalises large weights globally | Limits capacity in early epochs |
| Compatible with all architectures | Yes — drop-in at optimizer level | Yes — controlled by `head_only_epochs=0` to disable |
| Can be combined | Yes — A and B are orthogonal and additive | Yes |
| Expected effect on val loss | Slower train descent, val follows closer | Val loss starts lower (head-only is stable), then improves further |

### Recommendation

Implement Direction A first — it is a small, surgical change and addresses the most direct cause (train/val gap from large LR + no penalty). If val loss still diverges after that, add Direction B on top.

---

## Todo List

### Direction A — Weight Decay + ReduceLROnPlateau

- [x] **A.1** Add `weight_decay: float = typer.Option(1e-4, "-w", ...)` CLI parameter to `train()` in `typer_entrypoint.py`
- [x] **A.2** Add `weight_decay` to the `hyperparams` dict (for `hyperparams.json` logging)
- [x] **A.3** Pass `weight_decay=weight_decay` to `torch.optim.Adam(...)` in the fold loop
- [x] **A.4** Instantiate `ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)` after the optimizer
- [x] **A.5** Call `scheduler_step(epoch_val_loss)` at the end of each epoch, after `early_stopping()`
- [x] **A.6** Add `"weight_decay"` and `"lr_scheduler"` entries to `hyperparams` dict
- [x] **A.7** Run `uv run pyright src/` — expect 0 errors ✓

### Direction B — Phased Layer Unfreezing

- [x] **B.1** Add `head_only_epochs: int = typer.Option(3, "-H", ...)` CLI parameter to `train()` in `typer_entrypoint.py`
- [x] **B.2** Add `freeze_all: bool = False` parameter to `get_model()` in `regularized.py`
- [x] **B.3** In `regularized.get_model()`, wrap the layer4 unfreezing in `if not freeze_all:`
- [x] **B.4** In the fold loop, pass `freeze_all=(head_only_epochs > 0)` when constructing the REGULARIZED model
- [x] **B.5** Inside the epoch loop, add a phase transition block: when `epoch == head_only_epochs`, unfreeze layer4 via `optimizer.add_param_group()` (no optimizer/scheduler rebind needed)
- [x] **B.6** Add `"head_only_epochs"` to the `hyperparams` dict
- [x] **B.7** Run `uv run pyright src/` — expect 0 errors ✓
