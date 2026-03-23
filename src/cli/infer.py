"""
MIL inference and 5-panel attention heatmap visualisation for lampe-cli infer.

Loads a trained MIL model from an artifact directory, runs inference on a
single FOV, and saves a 5-panel PNG showing the three raw channels, a
pseudo-RGB composite, and an attention-weighted overlay.

All heavy imports are inside run() to preserve CLI lazy-startup behaviour.
"""


def run(artifact_dir: str, matpath: str, fov_index: int) -> None:
    """
    Run MIL inference on a single FOV and save the attention heatmap figure.

    Args:
        artifact_dir: Fold artifact directory (must contain model_weights.pth).
                      The parent directory must contain hyperparams.json.
        matpath:      Path to the Full images directory.
        fov_index:    Global FOV index to run inference on.

    Raises:
        FileNotFoundError: If hyperparams.json or model_weights.pth are missing.
        ValueError: If the artifact's architecture is not "mil", or if
                    fov_index is out of range.
    """
    # built-in
    import json
    import os
    from typing import Any, cast

    # third-party
    import numpy as np
    import scipy.ndimage
    import torch
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors
    from matplotlib import pyplot as plt

    # package
    from architectures import mil
    from shared.mat_reader import MatReader

    # --- Load and validate hyperparams ---
    parent_dir = os.path.dirname(os.path.abspath(artifact_dir))
    hyperparams_path = os.path.join(parent_dir, "hyperparams.json")
    if not os.path.isfile(hyperparams_path):
        raise FileNotFoundError(f"hyperparams.json not found at {hyperparams_path}")

    with open(hyperparams_path, "r", encoding="utf-8") as hf:
        hyperparams: dict[str, Any] = cast(dict[str, Any], json.load(hf))

    architecture_val: str = str(hyperparams.get("architecture", ""))
    if architecture_val != "mil":
        raise ValueError(
            f"lampe-cli infer only supports MIL artifacts. "
            f"Got architecture: {architecture_val!r}"
        )
    sliding_factor: int = int(hyperparams.get("sliding_factor", 5))

    # --- Dataset summary ---
    from cli.dataset_info import run as print_dataset_info

    mat_reader = MatReader(matpath)
    print_dataset_info(matpath)

    # --- Validate FOV index ---
    num_fovs = mat_reader.get_num_fovs()
    if not (0 <= fov_index < num_fovs):
        raise ValueError(f"--fov {fov_index} is out of range [0, {num_fovs - 1}]")

    class_names = ["Healthy", "LGC", "HGC", "IDC"]
    true_class = int(mat_reader.class_labels[fov_index])
    patient_id = str(mat_reader.patient_ids[fov_index])
    print(
        f"\nRunning inference on FOV {fov_index}  "
        f"(class: {class_names[true_class]}, patient: {patient_id})"
    )

    # --- Load model weights ---
    weights_path = os.path.join(artifact_dir, "model_weights.pth")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"No model weights found at {weights_path}. "
            "Run training with `-s` to save weights."
        )

    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = mil.get_model(num_classes=4, freeze_all=False).to(device)
    state_dict: dict[str, torch.Tensor] = torch.load(  # type: ignore[assignment]
        weights_path, map_location=device, weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()

    # --- Build MILDataset for the single FOV (val split: fixed grid, no aug) ---
    mil_dataset = mil.MILDataset(
        mat_reader,
        eff_fov_indices=[fov_index],
        factor=sliding_factor,
        train=False,
    )
    bag, _label, _pid = mil_dataset[0]
    bag = bag.to(device)

    # --- Run inference ---
    with torch.no_grad():
        mil_inf_out: tuple[torch.Tensor, torch.Tensor] = model(bag.unsqueeze(0))  # type: ignore[assignment]
        logits_batch, attn_batch = mil_inf_out

    logits = logits_batch.squeeze(0)  # (num_classes,)
    attn = attn_batch.squeeze(0)  # (N_patches,)

    probs_np: np.ndarray = torch.softmax(logits, dim=0).cpu().numpy()
    pred_class = int(logits.argmax().item())
    confidence = float(probs_np[pred_class])
    attn_np: np.ndarray = attn.cpu().numpy()

    # --- Build image panels ---
    height, width = mat_reader.get_height_width()
    raw_fov: np.ndarray = mat_reader.images[fov_index].astype(np.float32)  # (C, H, W)

    def _minmax(arr: np.ndarray) -> np.ndarray:
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo < 1e-8:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    ch0 = _minmax(raw_fov[0])  # SRS1 — Lipids   (1450 cm⁻¹)
    ch1 = _minmax(raw_fov[1])  # SRS2 — Proteins  (1668 cm⁻¹)
    ch2 = _minmax(raw_fov[2])  # SHG  — Collagen
    composite: np.ndarray = np.stack([ch0, ch1, ch2], axis=-1)  # (H, W, 3)

    # Attention heatmap: reshape (N,) → (factor, factor), upsample to (H, W)
    attn_grid: np.ndarray = attn_np.reshape(sliding_factor, sliding_factor)
    zoom_y = height / sliding_factor
    zoom_x = width / sliding_factor
    attn_upsampled: np.ndarray = scipy.ndimage.zoom(  # type: ignore[assignment]
        attn_grid, (zoom_y, zoom_x), order=1
    )
    attn_upsampled_norm: np.ndarray = _minmax(attn_upsampled)

    hot_cmap = mcm.get_cmap("hot")
    heatmap_rgba: np.ndarray = hot_cmap(attn_upsampled_norm)  # type: ignore[assignment]
    heatmap_rgb: np.ndarray = heatmap_rgba[:, :, :3]  # (H, W, 3)

    overlay: np.ndarray = np.clip(composite * 0.55 + heatmap_rgb * 0.45, 0.0, 1.0)

    # --- Render 1×5 figure ---
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    panel_titles = [
        "SRS1 \u2014 Lipids\n(1450 cm\u207b\u00b9)",
        "SRS2 \u2014 Proteins\n(1668 cm\u207b\u00b9)",
        "SHG \u2014 Collagen",
        "Composite\n(pseudo-RGB)",
        "Attention Overlay",
    ]

    axes[0].imshow(ch0, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title(panel_titles[0])
    axes[0].axis("off")

    axes[1].imshow(ch1, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title(panel_titles[1])
    axes[1].axis("off")

    axes[2].imshow(ch2, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].set_title(panel_titles[2])
    axes[2].axis("off")

    axes[3].imshow(composite)
    axes[3].set_title(panel_titles[3])
    axes[3].axis("off")

    axes[4].imshow(overlay)
    axes[4].set_title(panel_titles[4])
    axes[4].axis("off")

    # Colorbar showing raw attention weight scale
    norm = mcolors.Normalize(vmin=float(attn_np.min()), vmax=float(attn_np.max()))
    sm = mcm.ScalarMappable(norm=norm, cmap="hot")
    sm.set_array(np.array([]))
    cbar = fig.colorbar(sm, ax=axes[4], fraction=0.046, pad=0.04)
    cbar.set_label("Attention weight")

    suptitle = (
        f"FOV {fov_index}"
        f"  |  True: {class_names[true_class]} ({true_class})"
        f"  |  Predicted: {class_names[pred_class]} ({pred_class})"
        f"  |  Confidence: {confidence:.1%}"
    )
    fig.suptitle(suptitle, fontsize=12)

    plt.tight_layout()
    output_path = os.path.join(artifact_dir, f"inference_fov{fov_index}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")
