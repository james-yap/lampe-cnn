"""
Dataset summary utility for the lampe-cli dataset-info command.

Prints total FOV count, image dimensions, channel count, and a per-class
table of FOV counts, global index ranges, and patient IDs.

All heavy imports are inside run() to preserve CLI lazy-startup behaviour.
"""


def run(matpath: str) -> None:
    """
    Print a dataset summary for the given matpath directory.

    Args:
        matpath: Path to the Full images directory containing
                 {Class}_bulk_data.mat and {Class}_names.txt files.
    """
    import numpy as np
    from shared.mat_reader import MatReader
    from shared.constants import CLASS_NAMES

    mat_reader = MatReader(matpath)
    height, width = mat_reader.get_height_width()
    num_channels = mat_reader.get_num_channels()
    num_fovs = mat_reader.get_num_fovs()

    print(f"Dataset: {matpath}")
    print(
        f"Total FOVs : {num_fovs}   "
        f"Image size: {height} \u00d7 {width}   "
        f"Channels: {num_channels}"
    )
    print("Class distribution:")

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask: np.ndarray = mat_reader.class_labels == cls_idx
        indices_cls: np.ndarray = np.where(mask)[0]
        count = int(indices_cls.size)
        if count == 0:
            print(f"  {cls_idx}  {cls_name:<8}: {count:4d} FOVs")
            continue
        idx_min = int(indices_cls[0])
        idx_max = int(indices_cls[-1])
        patients: list[str] = sorted(
            set(str(p) for p in mat_reader.patient_ids[indices_cls])
        )
        patient_preview = ", ".join(patients[:5])
        if len(patients) > 5:
            patient_preview += f", ... ({len(patients)} total)"
        print(
            f"  {cls_idx}  {cls_name:<8}: {count:4d} FOVs   "
            f"indices [{idx_min} \u2013 {idx_max}]   "
            f"patients: {patient_preview}"
        )
