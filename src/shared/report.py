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
        train_preds: torch.Tensor | None = None,
        train_labels: torch.Tensor | None = None,
        train_pids: list[str] | None = None,
        val_pids: list[str] | None = None,
        is_ordinal: bool = False,
    ) -> None:
        """
        Decode raw model outputs, compute evaluation metrics, render a 2x3
        results figure, and save it to {output_dir}/results.png.

        Layout:
          [loss curve]  [train confusion matrix]  [val confusion matrix]
          [ROC curves]  [classification report (spans 2 columns)        ]

        Args:
            fold: 1-based fold number, used only for the figure title.
            output_dir: Directory in which to write results.png.
            train_losses: Per-epoch training loss values.
            val_losses: Per-epoch validation loss values.
            all_preds: Concatenated raw val model outputs.
                       Shape (N, num_classes) for standard architectures, or
                       (N, K-1) for ordinal architectures.
            all_labels: Concatenated integer val class labels, shape (N,).
            train_preds: Concatenated raw train model outputs from the last
                         epoch, same shape convention as all_preds. If None,
                         the train CM panel is left blank.
            train_labels: Concatenated integer train class labels from the
                          last epoch, shape (N,). If None, train CM is blank.
            train_pids: Patient IDs from the last training epoch, aligned
                        element-wise with train_labels. If None, train rows
                        are omitted from patient_splits.csv.
            val_pids: Patient IDs from the last validation epoch, aligned
                      element-wise with all_labels. If None, val rows are
                      omitted from patient_splits.csv.
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

        # --- Decode val predictions and reconstruct per-class probabilities ---
        pred_classes_np: np.ndarray
        all_probs: np.ndarray

        if is_ordinal:
            pred_classes_np = decode_ordinal(all_preds).numpy()
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

        # --- Decode train predictions (for train CM) ---
        train_pred_classes_np: np.ndarray | None = None
        if train_preds is not None and train_labels is not None:
            if is_ordinal:
                train_pred_classes_np = decode_ordinal(train_preds).numpy()
            else:
                train_pred_classes_np = train_preds.argmax(dim=1).numpy()

        # --- Val sklearn metrics ---
        cm_val = confusion_matrix(
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
        # sklearn returns (N, 1) for binary problems; expand to (N, 2) to keep
        # the ROC loop uniform across binary and multiclass cases.
        if self.num_classes == 2:
            all_labels_bin = np.hstack([1 - all_labels_bin, all_labels_bin])

        assert isinstance(all_labels_bin, np.ndarray) and isinstance(
            all_probs, np.ndarray
        ), (
            "Expected all_labels_bin and all_probs to be numpy arrays after "
            "label binarization and probability reconstruction, respectively."
        )

        # --- 2x3 mosaic figure ---
        # Row 0: loss curve | train CM | val CM
        # Row 1: ROC curves | classification report (spans 2 columns)
        fig, axes = plt.subplot_mosaic(
            [["loss", "train_cm", "val_cm"], ["roc", "report", "report"]],
            figsize=(21, 11),
        )
        fig.suptitle(f"Fold {fold} Evaluation", fontsize=14)

        # Loss curves
        axes["loss"].plot(train_losses, label="Train Loss")
        axes["loss"].plot(val_losses, label="Val Loss")
        axes["loss"].set_xlabel("Epoch")
        axes["loss"].set_ylabel("Loss")
        axes["loss"].set_title("Training and Validation Loss")
        axes["loss"].legend()

        # Train confusion matrix
        if train_pred_classes_np is not None and train_labels is not None:
            cm_train = confusion_matrix(
                train_labels.numpy(),
                train_pred_classes_np,
                labels=list(range(self.num_classes)),
            )
            ConfusionMatrixDisplay(confusion_matrix=cm_train).plot(
                ax=axes["train_cm"], colorbar=False
            )
            axes["train_cm"].set_title("Train Confusion Matrix")
        else:
            axes["train_cm"].axis("off")
            axes["train_cm"].set_title("Train Confusion Matrix (unavailable)")

        # Val confusion matrix
        ConfusionMatrixDisplay(confusion_matrix=cm_val).plot(
            ax=axes["val_cm"], colorbar=False
        )
        axes["val_cm"].set_title("Val Confusion Matrix")

        # ROC curves
        for i in range(self.num_classes):
            if all_labels_bin[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
            roc_auc = auc(fpr, tpr)
            axes["roc"].plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")
        axes["roc"].plot([0, 1], [0, 1], "k--", label="Random")
        axes["roc"].set_xlabel("False Positive Rate")
        axes["roc"].set_ylabel("True Positive Rate")
        axes["roc"].set_title("ROC Curves (Val)")
        axes["roc"].legend()

        # Classification report text
        axes["report"].axis("off")
        axes["report"].text(
            0.5,
            0.5,
            cr,
            fontsize=12,
            family="monospace",
            ha="center",
            va="center",
        )
        axes["report"].set_title("Classification Report (Val)")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "results.png"), dpi=150)
        plt.close()

        # --- Patient-split table (one row per unique patient per split) ---
        import csv

        seen_pids: set[tuple[str, str]] = set()  # (patient_id, split)
        rows: list[dict[str, str]] = []
        if val_pids is not None:
            for pid, lbl in zip(val_pids, all_labels_np):
                key = (pid, "val")
                if key in seen_pids:
                    continue
                seen_pids.add(key)
                lbl_int = int(lbl)
                rows.append(
                    {
                        "patient_id": pid,
                        "label": str(lbl_int),
                        "class_name": self.class_names[lbl_int],
                        "split": "val",
                    }
                )
        if train_pids is not None and train_labels is not None:
            train_labels_np: np.ndarray = train_labels.numpy()
            for pid, lbl in zip(train_pids, train_labels_np):
                key = (pid, "train")
                if key in seen_pids:
                    continue
                seen_pids.add(key)
                lbl_int = int(lbl)
                rows.append(
                    {
                        "patient_id": pid,
                        "label": str(lbl_int),
                        "class_name": self.class_names[lbl_int],
                        "split": "train",
                    }
                )
        with open(
            os.path.join(output_dir, "patient_splits.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file, fieldnames=["patient_id", "label", "class_name", "split"]
            )
            writer.writeheader()
            writer.writerows(rows)
