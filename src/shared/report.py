"""
Shared fold evaluation, reporting, and visualisation utility.

FoldReporter is stateless except for its num_classes / class_names configuration.
All heavy imports (matplotlib, sklearn, numpy) live inside save() so that importing
this module does not slow down the initial CLI response time.
"""

import torch


class FoldReporter:
    """
    Encapsulates all per-fold evaluation, metrics computation, and results.png
    generation so that the CLI training loop only has to call reporter.save(...).

    The is_ordinal flag in save() selects between two decode paths:
      - is_ordinal=True : decode_ordinal (sum of sigmoid thresholds) +
                          cumulative sigmoid probability reconstruction
      - is_ordinal=False: argmax class prediction + softmax probabilities
    """

    num_classes: int
    class_names: list[str]

    def __init__(self, num_classes: int, class_names: list[str]) -> None:
        self.num_classes = num_classes
        self.class_names = class_names

    def save(
        self,
        fold: int,
        output_dir: str,
        train_losses: list[float],
        val_losses: list[float],
        all_preds: torch.Tensor,
        all_labels: torch.Tensor,
        is_ordinal: bool = False,
    ) -> None:
        """
        Decode raw model outputs, compute evaluation metrics, render a 2x2
        results figure, and save it to {output_dir}/results.png.

        Args:
            fold: 1-based fold number, used only for the figure title.
            output_dir: Directory in which to write results.png.
            train_losses: Per-epoch training loss values.
            val_losses: Per-epoch validation loss values.
            all_preds: Concatenated raw model outputs.
                       Shape (N, num_classes) for standard architectures, or
                       (N, K-1) for ordinal architectures.
            all_labels: Concatenated integer class labels, shape (N,).
            is_ordinal: If True, uses the ordinal decode + cumulative-sigmoid
                        probability path. If False, uses argmax + softmax.
        """
        # Heavy imports kept here to preserve CLI lazy-startup behaviour
        import os
        import numpy as np
        from matplotlib import pyplot as plt
        from sklearn.metrics import (
            confusion_matrix,
            classification_report,
            ConfusionMatrixDisplay,
            roc_curve,
            auc,
        )
        from sklearn.preprocessing import label_binarize
        from architectures.ordinal import decode_ordinal

        all_labels_np: np.ndarray = all_labels.numpy()

        # --- Decode predictions and reconstruct per-class probabilities ---
        pred_classes_np: np.ndarray
        all_probs: np.ndarray

        if is_ordinal:
            pred_classes_np = decode_ordinal(all_preds).numpy()
            # Cumulative sigmoid probability reconstruction:
            #   P(class=0)   = 1 - σ(logit_0)
            #   P(class=k)   = σ(logit_{k-1}) - σ(logit_k)  for 0 < k < K-1
            #   P(class=K-1) = σ(logit_{K-2})
            sigm: np.ndarray = torch.sigmoid(all_preds).numpy()
            all_probs = np.concatenate(
                [
                    1.0 - sigm[:, 0:1],
                    sigm[:, 0:1] - sigm[:, 1:2],
                    sigm[:, 1:2] - sigm[:, 2:3],
                    sigm[:, 2:3],
                ],
                axis=1,
            )
        else:
            pred_classes_np = all_preds.argmax(dim=1).numpy()
            all_probs = torch.softmax(all_preds, dim=1).numpy()

        # --- Sklearn metrics ---
        cm = confusion_matrix(
            all_labels_np,
            pred_classes_np,
            labels=list(range(self.num_classes)),
        )
        cr: str = classification_report(
            all_labels_np,
            pred_classes_np,
            output_dict=False,
            zero_division=0,  # type: ignore[call-overload]
        )
        all_labels_bin: np.ndarray = label_binarize(
            all_labels_np, classes=list(range(self.num_classes))
        )

        assert isinstance(all_labels_bin, np.ndarray) and isinstance(
            all_probs, np.ndarray
        ), (
            "Expected all_labels_bin and all_probs to be numpy arrays after "
            "label binarization and probability reconstruction, respectively."
        )

        # --- 2x2 matplotlib figure ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        fig.suptitle(f"Fold {fold} Evaluation", fontsize=14)

        # Loss curves (top-left)
        axes[0, 0].plot(train_losses, label="Train Loss")
        axes[0, 0].plot(val_losses, label="Val Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training and Validation Loss")
        axes[0, 0].legend()

        # Confusion matrix (top-right)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=axes[0, 1], colorbar=False)
        axes[0, 1].set_title("Confusion Matrix")

        # ROC curves (bottom-left)
        for i in range(self.num_classes):
            if all_labels_bin[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
            roc_auc = auc(fpr, tpr)
            axes[1, 0].plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")
        axes[1, 0].plot([0, 1], [0, 1], "k--", label="Random")
        axes[1, 0].set_xlabel("False Positive Rate")
        axes[1, 0].set_ylabel("True Positive Rate")
        axes[1, 0].set_title("ROC Curves")
        axes[1, 0].legend()

        # Classification report text (bottom-right)
        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.5,
            0.5,
            cr,
            fontsize=12,
            family="monospace",
            ha="center",
            va="center",
        )
        axes[1, 1].set_title("Classification Report")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "results.png"), dpi=150)
        plt.close()
