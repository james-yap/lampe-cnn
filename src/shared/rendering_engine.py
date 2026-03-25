"""
FOV visualisation rendering for lampe-cli infer and lampe-cli view.

render_fov_figure() builds and saves the 5-panel mosaic figure used by both
commands.  When called from `infer`, attention data and a predicted class are
provided and the attention panel shows a per-patch heatmap with a contour
overlaid on the composite.  When called from `view` (no inference), the
attention panel is left blank and only the true label appears in the title.

Heavy imports (matplotlib, scipy) are inside render_fov_figure() to preserve
the CLI lazy-startup behaviour.
"""

import numpy as np


def render_fov_figure(
    raw_fov: np.ndarray,
    fov_index: int,
    true_class: int,
    class_names: list[str],
    output_path: str,
    attn_np: np.ndarray | None = None,
    sliding_factor: int = 5,
    pred_class: int | None = None,
    confidence: float | None = None,
) -> None:
    """
    Build and save a 5-panel mosaic figure for a single FOV.

    Layout (2-row mosaic)::

        AABBCC
        .DDEE.

        A — SRS1 (Lipids, 1450 cm⁻¹)   — Greens_r
        B — SRS2 (Proteins, 1668 cm⁻¹) — Reds_r
        C — SHG  (Collagen)             — Blues_r
        D — Composite pseudo-RGB (R=Proteins, G=Lipids, B=Collagen)
              + red contour at 75th-percentile attention boundary (when attn provided)
        E — Attention heatmap (hot colormap)  or  blank grey panel

    Channel normalisation: robust_minmax + gamma correction (gamma=1.5 for SRS,
    1.2 for SHG).

    Args:
        raw_fov:       Raw FOV image array, shape (C, H, W), float32.
        fov_index:     Global FOV index used in the suptitle.
        true_class:    Integer true class label.
        class_names:   List mapping integer class index to display name.
        output_path:   Full path at which to save the PNG.
        attn_np:       Per-patch attention weights, shape (N_patches,).  If
                       None the attention panel is rendered as a blank grey
                       image and no contour is drawn on the composite.
        sliding_factor: Spatial grid dimension (factor×factor = N_patches).
                        Used to reshape and upsample attn_np.  Ignored when
                        attn_np is None.
        pred_class:    Predicted class index.  If None the suptitle shows
                       only the true label (view mode).
        confidence:    Softmax probability of the predicted class.  Only
                       shown in the suptitle when pred_class is also provided.
    """
    import scipy.ndimage
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors
    from matplotlib import pyplot as plt
    from shared.utils import robust_minmax

    height, width = raw_fov.shape[1], raw_fov.shape[2]

    # --- Channel normalisation + gamma ---
    gamma = 1.5
    ch0: np.ndarray = robust_minmax(raw_fov[0]) ** gamma          # Lipids
    ch1: np.ndarray = robust_minmax(raw_fov[1]) ** gamma          # Proteins
    ch2: np.ndarray = robust_minmax(raw_fov[2], p_max=99.9) ** 1.2  # Collagen

    # R=Proteins, G=Lipids, B=Collagen
    composite: np.ndarray = np.stack([ch1, ch0, ch2], axis=-1)  # (H, W, 3)

    # --- Attention heatmap ---
    attn_upsampled_norm: np.ndarray | None = None
    heatmap_rgb: np.ndarray | None = None

    if attn_np is not None:
        attn_grid: np.ndarray = attn_np.reshape(sliding_factor, sliding_factor)
        zoom_y = height / sliding_factor
        zoom_x = width / sliding_factor
        attn_upsampled: np.ndarray = scipy.ndimage.zoom(  # type: ignore[assignment]
            attn_grid, (zoom_y, zoom_x), order=1
        )
        attn_upsampled_norm = robust_minmax(attn_upsampled)
        hot_cmap = mcm.get_cmap("hot")
        heatmap_rgba: np.ndarray = hot_cmap(attn_upsampled_norm)  # type: ignore[assignment]
        heatmap_rgb = heatmap_rgba[:, :, :3]  # (H, W, 3)

    # --- Mosaic layout ---
    layout = """
    AABBCC
    .DDEE.
    """
    fig, axes = plt.subplot_mosaic(layout, figsize=(15, 10))

    axes["A"].imshow(ch0, cmap="Greens_r", vmin=0.0, vmax=1.0)
    axes["A"].set_title("SRS1 \u2014 Lipids\n(1450 cm\u207b\u00b9)")
    axes["A"].axis("off")

    axes["B"].imshow(ch1, cmap="Reds_r", vmin=0.0, vmax=1.0)
    axes["B"].set_title("SRS2 \u2014 Proteins\n(1668 cm\u207b\u00b9)")
    axes["B"].axis("off")

    axes["C"].imshow(ch2, cmap="Blues_r", vmin=0.0, vmax=1.0)
    axes["C"].set_title("SHG \u2014 Collagen")
    axes["C"].axis("off")

    axes["D"].imshow(composite)
    axes["D"].set_title("Composite\n(pseudo-RGB)")
    axes["D"].axis("off")

    if attn_upsampled_norm is not None:
        threshold = float(np.percentile(attn_upsampled_norm, 75))
        axes["D"].contour(
            attn_upsampled_norm, levels=[threshold], colors="red", linewidths=3
        )

    if heatmap_rgb is not None:
        axes["E"].imshow(heatmap_rgb)
        axes["E"].set_title("Attention Heatmap")
        axes["E"].axis("off")
        if attn_np is not None:
            norm = mcolors.Normalize(
                vmin=float(attn_np.min()), vmax=float(attn_np.max())
            )
            sm = mcm.ScalarMappable(norm=norm, cmap="hot")
            sm.set_array(np.array([]))
            cbar = fig.colorbar(sm, ax=axes["E"], fraction=0.046, pad=0.04)
            cbar.set_label("Attention weight")
    else:
        axes["E"].imshow(np.zeros((height, width)), cmap="gray", vmin=0.0, vmax=1.0)
        axes["E"].set_title("Attention Heatmap\n(no inference)")
        axes["E"].axis("off")

    # --- Suptitle ---
    true_name = class_names[true_class]
    if pred_class is not None and confidence is not None:
        pred_name = class_names[pred_class]
        suptitle = (
            f"FOV {fov_index}"
            f"  |  True: {true_name} ({true_class})"
            f"  |  Predicted: {pred_name} ({pred_class})"
            f"  |  Confidence: {confidence:.1%}"
        )
    else:
        suptitle = f"FOV {fov_index}  |  True: {true_name} ({true_class})"

    fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
