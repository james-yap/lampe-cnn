"""
MIL inference for lampe-cli infer.

Handles model loading, forward pass, and result extraction.  Figure rendering
is delegated to shared.rendering_engine.render_fov_figure().

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
    import torch

    # package
    from architectures import mil
    from shared.mat_reader import MatReader
    from shared.rendering_engine import render_fov_figure

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

    # --- Render and save ---
    output_path = os.path.join(artifact_dir, f"inference_fov{fov_index}.png")
    render_fov_figure(
        raw_fov=mat_reader.images[fov_index].astype(np.float32),
        fov_index=fov_index,
        true_class=true_class,
        class_names=class_names,
        output_path=output_path,
        attn_np=attn_np,
        sliding_factor=sliding_factor,
        pred_class=pred_class,
        confidence=confidence,
    )
    print(f"Saved: {output_path}")
