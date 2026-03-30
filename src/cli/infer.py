"""
LinearSVM inference with LayerCAM for lampe-cli infer.

Handles model loading, forward pass, LayerCAM heatmap generation, and figure
rendering.  Figure rendering is delegated to shared.rendering_engine.render_fov_figure().

LayerCAM (Jiang et al., 2021) hooks into the target convolutional layer and
computes element-wise products of the activation maps with their local
gradient weights, then sums across channels and applies ReLU.  This produces
a pixel-level relevance map at the spatial resolution of the target layer,
which is then upsampled to the full FOV dimensions for overlay.

Target layer: feature_extractor.layer2  (last layer of ResNet18 before global
pool).  Output shape: (B, 128, H/8, W/8) — spatially informative at ~8×
downsampling.

All heavy imports are inside run() to preserve CLI lazy-startup behaviour.
"""


def run(artifact_dir: str, matpath: str, fov_index: int) -> None:
    """
    Run LSVM inference on a single FOV and save a LayerCAM heatmap figure.

    Args:
        artifact_dir: Fold artifact directory (must contain model_weights.pth).
                      The parent directory must contain hyperparams.json.
        matpath:      Path to the Full images directory.
        fov_index:    Global FOV index to run inference on.

    Raises:
        FileNotFoundError: If hyperparams.json or model_weights.pth are missing.
        ValueError: If the artifact's architecture is not "lsvm", or if
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

    # package
    from architectures import linear_svm
    from shared.mat_reader import MatReader
    from shared.rendering_engine import render_fov_figure
    from shared.constants import CLASS_NAMES

    # --- Load and validate hyperparams ---
    parent_dir = os.path.dirname(os.path.abspath(artifact_dir))
    hyperparams_path = os.path.join(parent_dir, "hyperparams.json")
    if not os.path.isfile(hyperparams_path):
        raise FileNotFoundError(f"hyperparams.json not found at {hyperparams_path}")

    with open(hyperparams_path, "r", encoding="utf-8") as hf:
        hyperparams: dict[str, Any] = cast(dict[str, Any], json.load(hf))

    architecture_val: str = str(hyperparams.get("architecture", ""))
    if architecture_val != "lsvm":
        raise ValueError(
            f"lampe-cli infer only supports lsvm artifacts. "
            f"Got architecture: {architecture_val!r}"
        )

    # --- Dataset summary ---
    from cli.dataset_info import run as print_dataset_info

    mat_reader = MatReader(matpath)
    print_dataset_info(matpath)

    # --- Validate FOV index ---
    num_fovs = mat_reader.get_num_fovs()
    if not (0 <= fov_index < num_fovs):
        raise ValueError(f"--fov {fov_index} is out of range [0, {num_fovs - 1}]")

    class_names = CLASS_NAMES
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

    state_dict: dict[str, torch.Tensor] = torch.load(  # type: ignore[assignment]
        weights_path, map_location=device, weights_only=True
    )
    # Derive num_classes from the saved fc.weight rather than assuming 4,
    # so weights trained on any class count load correctly.
    num_classes: int = int(state_dict["fc.weight"].shape[0])
    model = linear_svm.get_model(num_classes=num_classes, freeze_all=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # --- Prepare input tensor ---
    raw_fov: np.ndarray = mat_reader.images[fov_index].astype(np.float32)  # (C, H, W)
    height, width = raw_fov.shape[1], raw_fov.shape[2]

    input_tensor = torch.from_numpy(raw_fov).unsqueeze(0).to(device)  # (1, C, H, W)
    input_tensor.requires_grad_(
        True
    )  # seed the computation graph so feat_map tracks grads

    # --- Forward pass with intermediate capture for LayerCAM ---
    # Hooks on submodules inside a create_feature_extractor GraphModule are
    # unreliable: torch.fx may not emit a single call_module node for layer2,
    # so hooks never fire.  Instead, replicate LinearSVM.forward() manually
    # and call retain_grad() on the layer2 feature map tensor directly.
    #
    # LayerCAM formula: CAM = ReLU( sum_c( A_c * dScore/dA_c ) )
    from architectures.linear_svm import LinearSVM

    lsvm = cast(LinearSVM, model)

    feat_map: torch.Tensor = lsvm.feature_extractor(input_tensor)[
        "features"
    ]  # (1, 128, H/8, W/8)
    feat_map.retain_grad()  # keep grad on this non-leaf tensor
    out: torch.Tensor = lsvm.fc(torch.flatten(lsvm.pool(feat_map), 1))

    pred_class = int(out.squeeze(0).argmax().item())
    probs_np: np.ndarray = torch.softmax(out.squeeze(0), dim=0).detach().cpu().numpy()
    confidence = float(probs_np[pred_class])

    print(
        f"Predicted: {class_names[pred_class]} ({pred_class})  "
        f"| Confidence: {confidence:.1%}"
    )

    # --- Generate LayerCAM ---
    out[0, pred_class].backward()

    assert feat_map.grad is not None
    cam = torch.relu((feat_map * feat_map.grad).sum(dim=1)).squeeze(0)  # (H_cam, W_cam)
    cam_np: np.ndarray = cam.detach().cpu().numpy()

    # Upsample from layer2 spatial resolution (H/8 × W/8) to full FOV
    zoom_y = height / cam_np.shape[0]
    zoom_x = width / cam_np.shape[1]
    cam_upsampled: np.ndarray = scipy.ndimage.zoom(cam_np, (zoom_y, zoom_x), order=1)  # type: ignore[assignment]

    # Normalize to [0, 1] using robust percentiles to avoid saturation from
    # outlier activations — clip at 2nd/98th percentile before rescaling.
    p_low = float(np.percentile(cam_upsampled, 2))
    p_high = float(np.percentile(cam_upsampled, 98))
    if p_high > p_low:
        cam_norm: np.ndarray = np.clip(
            (cam_upsampled - p_low) / (p_high - p_low), 0.0, 1.0
        )
    else:
        cam_norm = np.zeros_like(cam_upsampled)

    # --- Render and save ---
    output_path = os.path.join(artifact_dir, f"inference_fov{fov_index}.png")
    render_fov_figure(
        raw_fov=raw_fov,
        fov_index=fov_index,
        true_class=true_class,
        class_names=class_names,
        output_path=output_path,
        heatmap_2d=cam_norm,
        pred_class=pred_class,
        confidence=confidence,
    )
    print(f"Saved: {output_path}")
