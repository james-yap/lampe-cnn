"""
FOV visualisation for lampe-cli view.

Renders the 5-panel mosaic figure for a single FOV without running inference.
The attention panel is left blank and only the true label appears in the title.

All heavy imports are inside run() to preserve CLI lazy-startup behaviour.
"""

from shared.constants import CLASS_NAMES


def run(matpath: str, fov_index: int, output_dir: str = ".") -> None:
    """
    Render a 5-panel FOV figure (no inference) and save it to output_dir.

    Args:
        matpath:    Path to the Full images directory.
        fov_index:  Global FOV index to visualise.
        output_dir: Directory in which to save the output PNG.
                    Defaults to the current working directory.

    Raises:
        ValueError: If fov_index is out of range.
    """
    import os
    import numpy as np
    from shared.mat_reader import MatReader
    from shared.rendering_engine import render_fov_figure
    from lampe_cli.dataset_info import run as print_dataset_info

    mat_reader = MatReader(matpath)
    print_dataset_info(matpath)

    num_fovs = mat_reader.get_num_fovs()
    if not (0 <= fov_index < num_fovs):
        raise ValueError(f"--fov {fov_index} is out of range [0, {num_fovs - 1}]")

    true_class = int(mat_reader.class_labels[fov_index])
    patient_id = str(mat_reader.patient_ids[fov_index])
    print(
        f"\nViewing FOV {fov_index}  "
        f"(class: {CLASS_NAMES[true_class]}, patient: {patient_id})"
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"view_fov{fov_index}.png")

    render_fov_figure(
        raw_fov=mat_reader.images[fov_index].astype(np.float32),
        fov_index=fov_index,
        true_class=true_class,
        class_names=CLASS_NAMES,
        output_path=output_path,
    )
    print(f"Saved: {output_path}")
