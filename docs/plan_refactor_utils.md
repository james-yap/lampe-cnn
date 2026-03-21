# Plan: Refactor — Shared Reporting Utility and Optimizer Engine

## Motivation

`typer_entrypoint.py` currently has two large blocks of inline code that are not
directly related to the training control flow:

### Problem 1 — Inline fold reporting (~85 lines inside the fold loop)

After every fold, the CLI manually:
- Decodes predictions (ordinal inverse-cumulative sigmoid vs. argmax)
- reconstructs 4-class probabilities from K-1 logits
- calls `confusion_matrix`, `classification_report`, `label_binarize`
- builds a 2×2 matplotlib figure (loss curve, confusion matrix, ROC, report text)
- saves `results.png`

This block has no separation of concerns: display logic, probability math, and file I/O
are all mixed together inside an `if all_preds and all_labels:` guard. Adding a new
architecture (e.g. MIL) requires editing this block again and duplicating the conditional
decode logic. Adding a new plot (e.g. attention heatmap) pollutes the already long CLI
function.

### Problem 2 — Scattered optimizer assembly (~25 lines spread over the fold and epoch loops)

The training loop currently:
- selects `BCEWithLogitsLoss` vs. `CrossEntropyLoss` based on `architecture`
- constructs Adam with `weight_decay` (REGULARIZED) or without (all others), and with
  a `requires_grad` param filter (REGULARIZED) or `model.parameters()` (all others)
- instantiates a `ReduceLROnPlateau` scheduler and wraps it as a `Callable` to avoid
  a pyright issue
- inside the epoch loop, checks `epoch == head_only_epochs` and calls
  `optimizer.add_param_group(...)` to inject layer4

These concerns are scattered across two scopes (fold-level and epoch-level), making it
hard to see at a glance which architectures use which training strategy, and making it
error-prone to add new strategies (every new architecture must be handled in both
scopes).

---

## Design

### `shared/report.py` — `FoldReporter`

A stateless helper that owns all reporting and visualization logic.

```python
class FoldReporter:
    num_classes: int
    class_names: list[str]

    def __init__(self, num_classes: int, class_names: list[str]) -> None: ...

    def save(
        self,
        fold: int,
        output_dir: str,
        train_losses: list[float],
        val_losses: list[float],
        all_preds: torch.Tensor,   # (N, num_classes) or (N, K-1) depending on is_ordinal
        all_labels: torch.Tensor,  # (N,) integer class labels
        is_ordinal: bool = False,
    ) -> None: ...
```

Internally `save()` does, in order:

1. **Decode predictions to integer classes**:
   - `is_ordinal=True`: `decode_ordinal(all_preds)` from `architectures.ordinal`
   - `is_ordinal=False`: `all_preds.argmax(dim=1)`
2. **Reconstruct per-class probability estimates**:
   - `is_ordinal=True`: cumulative sigmoid differences (P(k) = σ(logit_{k-1}) − σ(logit_k))
   - `is_ordinal=False`: `torch.softmax(all_preds, dim=1)`
3. **Compute metrics**: `confusion_matrix`, `classification_report`, `label_binarize`
4. **Build 2×2 figure**: loss curves (top-left), confusion matrix (top-right), ROC curves (bottom-left), report text (bottom-right)
5. **Save** to `{output_dir}/results.png`, then close the figure

All heavy imports (`matplotlib`, `sklearn`, `numpy`) remain inside `save()` to preserve
the lazy-import strategy for CLI startup time.

The `is_ordinal` flag is intentionally a simple boolean rather than an architecture enum
reference so that `FoldReporter` has no knowledge of which architectures exist — it only
knows how to interpret two kinds of raw model outputs. New architectures that are neither
ordinal nor standard (e.g. MIL) will use `is_ordinal=False`, as their `forward()` directly
returns 4 class logits.

### `shared/optimizer_engine.py` — `OptimizerEngine`

An object that encapsulates all optimizer, scheduler, and phased-unfreezing state for
one training run.

```python
class OptimizerEngine:
    criterion: torch.nn.Module
    optimizer: torch.optim.Optimizer

    def __init__(
        self,
        model: torch.nn.Module,
        lr: float,
        weight_decay: float,
        head_only_epochs: int,
        use_ordinal_loss: bool,
        use_weight_decay: bool,
        use_scheduler: bool,
        use_phased_unfreezing: bool,
    ) -> None: ...

    @classmethod
    def for_architecture(
        cls,
        architecture: "Architecture",       # from typer_entrypoint
        model: torch.nn.Module,
        lr: float,
        weight_decay: float,
        head_only_epochs: int,
    ) -> "OptimizerEngine": ...

    def step_scheduler(self, val_loss: float) -> None:
        """Call once per epoch after computing val_loss. No-op if no scheduler."""

    def maybe_transition_phase(self, epoch: int) -> bool:
        """
        Call at the start of each epoch. If phase 2 threshold is reached,
        unfreezes layer4 and adds a new param group to the optimizer.
        Returns True on the epoch the transition fires, False otherwise.
        """
```

#### `for_architecture` flag mapping

| Architecture | `use_ordinal_loss` | `use_weight_decay` | `use_scheduler` | `use_phased_unfreezing` |
|---|---|---|---|---|
| `SLIDING_WINDOW` | False | False | False | False |
| `STANDARDIZED` | False | False | False | False |
| `CLASS_BALANCED` | False | False | False | False |
| `ORDINAL` | True | False | False | False |
| `REGULARIZED` | True | True | True | True |

New architectures only need a new row in `for_architecture`.

#### Internal init logic

```python
# criterion
self.criterion = BCEWithLogitsLoss() if use_ordinal_loss else CrossEntropyLoss()

# optimizer (filter to requires_grad=True params if we will be adding a group later)
params = (
    [p for p in model.parameters() if p.requires_grad]
    if use_phased_unfreezing
    else list(model.parameters())
)
optim_kwargs: dict[str, Any] = {"lr": lr}
if use_weight_decay:
    optim_kwargs["weight_decay"] = weight_decay
self.optimizer = torch.optim.Adam(params, **optim_kwargs)

# scheduler — stored as Callable to avoid pyright's self-referential type issue
self._scheduler_step: Callable[[float], None] | None = None
if use_scheduler:
    _s = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
    self._scheduler_step = _s.step

# phased unfreezing state
self._use_phased_unfreezing = use_phased_unfreezing
self._head_only_epochs = head_only_epochs
self._phase_transitioned = False
self._model = model
self._lr = lr
self._weight_decay = weight_decay
```

#### `maybe_transition_phase` logic

```python
def maybe_transition_phase(self, epoch: int) -> bool:
    if not self._use_phased_unfreezing or self._phase_transitioned:
        return False
    if epoch != self._head_only_epochs:
        return False
    layer4_params = [p for name, p in self._model.named_parameters() if "layer4" in name]
    for p in layer4_params:
        p.requires_grad = True
    self.optimizer.add_param_group(
        {"params": layer4_params, "lr": self._lr * 0.1, "weight_decay": self._weight_decay}
    )
    self._phase_transitioned = True
    return True
```

---

## Changes Required

### New files

- `src/shared/report.py` — `FoldReporter` class
- `src/shared/optimizer_engine.py` — `OptimizerEngine` class

### `src/cli/typer_entrypoint.py`

- Import `FoldReporter` from `shared.report` and `OptimizerEngine` from `shared.optimizer_engine` in the lazy import block
- Remove internal `criterion`/`optimizer`/`scheduler_step` construction (replaced by `OptimizerEngine.for_architecture(...)`)
- Remove the inline phase-transition block inside the epoch loop (replaced by `engine.maybe_transition_phase(epoch)`)
- Remove the inline scheduler step call (replaced by `engine.step_scheduler(epoch_val_loss)`)
- Create a single `FoldReporter(NUM_CLASSES, CLASS_NAMES)` before the fold loop
- Replace the `if all_preds and all_labels:` reporting block with `reporter.save(...)`
- Pass `is_ordinal=architecture in (Architecture.ORDINAL, Architecture.REGULARIZED)` to `reporter.save()`
- Remove lazy imports that are now owned by `report.py`: `confusion_matrix`, `classification_report`, `ConfusionMatrixDisplay`, `roc_curve`, `auc`, `label_binarize`, `plt`

### `src/shared/__init__.py`

No changes needed (existing empty init is fine; consumers import with full path).

### `docs/research.md`

- Update Section 5 (Shared Utilities) with `FoldReporter` and `OptimizerEngine` subsections
- Update Section 7 (CLI) training loop description to reflect delegation to both utilities
- Update Section 14 (Summary Diagram) to show the new shared utilities in the pipeline

---

## Todo List

### `FoldReporter`

- [ ] **R.1** Create `src/shared/report.py`
- [ ] **R.2** Implement `FoldReporter.__init__(self, num_classes, class_names)`
- [ ] **R.3** Implement `FoldReporter.save(...)` — full decode → metrics → figure → save pipeline
  - [ ] **R.3a** Ordinal decode path: `decode_ordinal` + cumulative sigmoid probs
  - [ ] **R.3b** Standard decode path: argmax + softmax probs
  - [ ] **R.3c** Metrics: `confusion_matrix`, `classification_report`, `label_binarize`
  - [ ] **R.3d** 2×2 matplotlib figure: loss curve, confusion matrix, ROC curves, report text
  - [ ] **R.3e** `plt.savefig` + `plt.close()`
- [ ] **R.4** Add `report.py` to `SOURCES.txt` (or verify editable install picks it up automatically)

### `OptimizerEngine`

- [ ] **O.1** Create `src/shared/optimizer_engine.py`
- [ ] **O.2** Implement `OptimizerEngine.__init__(...)` with all flag-driven logic
- [ ] **O.3** Implement `OptimizerEngine.for_architecture(cls, architecture, model, lr, weight_decay, head_only_epochs)` classmethod with the 5-architecture flag table
- [ ] **O.4** Implement `OptimizerEngine.step_scheduler(val_loss)` — no-op if no scheduler
- [ ] **O.5** Implement `OptimizerEngine.maybe_transition_phase(epoch) -> bool` — unfreeze + add_param_group once

### CLI Refactor

- [ ] **C.1** Add `FoldReporter` and `OptimizerEngine` to the lazy import block in `train()`
- [ ] **C.2** Remove the existing criterion / optimizer / scheduler construction lines
- [ ] **C.3** Replace with `engine = OptimizerEngine.for_architecture(...)`; use `engine.criterion` and `engine.optimizer`
- [ ] **C.4** Replace the phase-transition `if` block in the epoch loop with `engine.maybe_transition_phase(epoch)`
- [ ] **C.5** Replace `if scheduler_step is not None: scheduler_step(val_loss)` with `engine.step_scheduler(val_loss)`
- [ ] **C.6** Instantiate `reporter = FoldReporter(NUM_CLASSES, CLASS_NAMES)` before the fold loop
- [ ] **C.7** Replace the `if all_preds and all_labels:` reporting block with `reporter.save(...)`
- [ ] **C.8** Remove now-unused lazy imports: `confusion_matrix`, `classification_report`, `ConfusionMatrixDisplay`, `roc_curve`, `auc`, `label_binarize`, `plt`, `Callable`

### Validation

- [ ] **V.1** Run `uv run pyright src/` — expect 0 errors
- [ ] **V.2** Smoke run `lampe-cli train regularized <matpath> -e 3` — verify `results.png` is generated and phase transition prints
- [ ] **V.3** Verify `results.png` layout is identical to the pre-refactor output
