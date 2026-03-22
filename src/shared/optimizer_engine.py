"""
Shared optimizer, criterion, scheduler, and phased-unfreezing engine.

OptimizerEngine encapsulates all training-configuration concerns that vary per
architecture, so the CLI fold loop only calls:

    engine = OptimizerEngine.for_architecture(architecture, model, lr, wd, hoe)
    ...
    if engine.maybe_transition_phase(epoch):
        print("[Phase 2] Unfreezing layer4")
    ...
    engine.step_scheduler(epoch_val_loss)

The for_architecture() classmethod translates an Architecture enum value into
a set of boolean capability flags, keeping every architecture-specific branch in
one place.  Adding a new architecture requires only a new row in that method.
"""

from collections.abc import Callable

import torch
from torch import nn


class OptimizerEngine:
    """
    Architecture-aware container for criterion, optimizer, optional LR scheduler,
    and optional phased layer-unfreezing logic.

    Public attributes consumed by the training loop:
        criterion  — loss function (BCEWithLogitsLoss or CrossEntropyLoss)
        optimizer  — Adam optimizer, correctly configured for this architecture
    """

    criterion: nn.Module
    optimizer: torch.optim.Optimizer

    def __init__(
        self,
        model: nn.Module,
        lr: float,
        weight_decay: float,
        head_only_epochs: int,
        use_ordinal_loss: bool,
        use_weight_decay: bool,
        use_scheduler: bool,
        use_phased_unfreezing: bool,
    ) -> None:
        # --- Criterion ---
        self.criterion = (
            nn.BCEWithLogitsLoss() if use_ordinal_loss else nn.CrossEntropyLoss()
        )

        # --- Optimizer ---
        # When phased unfreezing is active the model starts with layer4 frozen, so
        # only the head params are requires_grad=True.  We must NOT add layer4 to
        # the optimizer here - they are injected later via add_param_group().
        initial_params = (
            [p for p in model.parameters() if p.requires_grad]
            if use_phased_unfreezing
            else list(model.parameters())
        )
        if use_weight_decay:
            self.optimizer = torch.optim.Adam(
                initial_params, lr=lr, weight_decay=weight_decay
            )
        else:
            self.optimizer = torch.optim.Adam(initial_params, lr=lr)

        # --- Scheduler stored as a plain Callable to avoid pyright's
        #     self-referential type issue in ReduceLROnPlateau stubs ---
        self._scheduler_step: Callable[[float], None] | None = None
        if use_scheduler:
            _s = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
            )
            self._scheduler_step = _s.step

        # --- Phased unfreezing state ---
        self._use_phased_unfreezing = use_phased_unfreezing
        self._head_only_epochs = head_only_epochs
        self._phase_transitioned = False
        self._model = model
        self._lr = lr
        self._weight_decay = weight_decay if use_weight_decay else 0.0

    @classmethod
    def for_architecture(
        cls,
        architecture: str,
        model: nn.Module,
        lr: float,
        weight_decay: float,
        head_only_epochs: int,
    ) -> "OptimizerEngine":
        """
        Construct an OptimizerEngine from an architecture name string.

        Flag mapping:

          architecture     ordinal_loss  weight_decay  scheduler  phased_unfreeze
          ---------------  ------------ ------------- ---------  ---------------
          sliding_window   False        False         False      False
          standardized     False        False         False      False
          class_balanced   False        False         False      False
          ordinal          True         False         False      False
          regularized      True         True          True       True
        """
        use_ordinal_loss = architecture in ("ordinal", "regularized")
        use_weight_decay = architecture == "regularized"
        use_scheduler = architecture == "regularized"
        use_phased_unfreezing = architecture == "regularized"

        return cls(
            model=model,
            lr=lr,
            weight_decay=weight_decay,
            head_only_epochs=head_only_epochs,
            use_ordinal_loss=use_ordinal_loss,
            use_weight_decay=use_weight_decay,
            use_scheduler=use_scheduler,
            use_phased_unfreezing=use_phased_unfreezing,
        )

    def step_scheduler(self, val_loss: float) -> None:
        """
        Step the LR scheduler with the current validation loss.
        No-op if this architecture has no scheduler.
        """
        if self._scheduler_step is not None:
            self._scheduler_step(val_loss)

    def maybe_transition_phase(self, epoch: int) -> bool:
        """
        If phased unfreezing is active and epoch equals head_only_epochs,
        mark all layer4 parameters as trainable and inject them into the
        optimizer as a new param group with a lower learning rate.

        Returns True on the epoch the transition fires, False every other epoch.
        """
        if not self._use_phased_unfreezing or self._phase_transitioned:
            return False
        if epoch != self._head_only_epochs:
            return False

        layer4_params = [
            p for name, p in self._model.named_parameters() if "layer4" in name
        ]
        for p in layer4_params:
            p.requires_grad = True
        self.optimizer.add_param_group(
            {
                "params": layer4_params,
                "lr": self._lr * 0.1,
                "weight_decay": self._weight_decay,
            }
        )
        self._phase_transitioned = True
        return True
