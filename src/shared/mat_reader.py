"""
MatReader class implementation.
"""

import os
from typing import Any, cast
import numpy as np
from scipy.io import loadmat # type: ignore

class MatReader:
    """
    Takes in a directory path and extracts .mat files by class.
    Unified interface for loading LAMPE datasets into PyTorch DataLoaders.
    Automatically groups data by patient to prevent intracore bias.
    """

    # shape: (n patients, np.ndarray(n samples per patient, width, height, n modalities))
    data: list[np.ndarray]

    def __init__(self, matpath: str):
        classes = ["Healthy", "HGC", "IDC", "LGC"]
        modalities = ["1450_bgsub", "1668_bgsub", "SHG"]

        # Verify files exist
        for classname in classes:
            mat_filename = f"{classname}_bulk_data.mat"
            if not os.path.isfile(os.path.join(matpath, mat_filename)):
                raise FileNotFoundError(f"Expected file '{mat_filename}' not found in path '{matpath}'")

            names_filename = f"{classname}_names.txt"
            if not os.path.isfile(os.path.join(matpath, names_filename)):
                raise FileNotFoundError(f"Expected file '{names_filename}' not found in path '{matpath}'")

        for classname in classes:
            mat_filename = f"{classname}_bulk_data.mat"
            names_filename = f"{classname}_names.txt"

            raw_data: dict[str, Any] = cast(dict[str, Any], loadmat(os.path.join(matpath, mat_filename)))

            with open(os.path.join(matpath, names_filename), 'r', encoding='utf-8') as f:
                # format: "1 B1 IDC 2": Slide number, position on slide, class, fov number
                names = [line.strip().split() for line in f.readlines()]

                mode_data_images: list[np.ndarray] = []

                for mode in modalities:
                    mode_key = f"{classname}_{mode}"
                    mode_data = np.asarray(cast(np.ndarray, raw_data[mode_key]))

                    if mode_data.ndim < 3:
                        raise ValueError(
                            f"Expected 3D array for '{mode_key}', got shape {mode_data.shape}"
                        )

                    assert len(names) == int(mode_data.shape[2]), \
                          f"Number of names in '{names_filename}' does not match number of samples in '{mat_filename}' for modality '{mode}'"

                    mode_data_images.append(mode_data) # (width, height, n samples)

                multimodal_data = np.stack(mode_data_images, axis=-1) # (width, height, n samples, n modalities)
                multimodal_data = np.transpose(multimodal_data, (2, 0, 1, 3)) # (n samples, width, height, n modalities)
                
                stratified_images: dict[str, list[np.ndarray]] = {}
                for multimodal_image, [slide_num, slide_pos, _class, _fov_num] in zip(multimodal_data, names):
                    patient_id = f"{slide_num}_{slide_pos}"
                    if patient_id not in stratified_images:
                        stratified_images[patient_id] = []
                    stratified_images[patient_id].append(multimodal_image)
                
                self.data = [np.stack(images, axis=0) for images in stratified_images.values()] # list of (n samples per patient, width, height, n modalities)

    def get_data(self) -> list[np.ndarray]:
        """
        Returns the loaded data as a list of numpy arrays,
        where each array corresponds to a patient and has shape (n samples per patient, width, height, n modalities).
        """
        return self.data