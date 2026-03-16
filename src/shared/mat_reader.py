"""
MatReader class implementation.
"""

import os
from typing import Any, cast
import numpy as np
from scipy.io import loadmat  # type: ignore


class MatReader:
    """
    Takes in a directory path and extracts .mat files by class.
    Unified interface for loading LAMPE datasets into PyTorch DataLoaders.
    Automatically groups data by patient to prevent intracore bias.
    """

    # shape: np.ndarray(n samples, n modalities, width, height)
    images: np.ndarray

    # shape: np.ndarray(n samples,)
    # 0: Healthy, 1: HGC, 2: IDC, 3: LGC
    class_labels: np.ndarray

    # shape: np.ndarray(n samples,)
    patient_ids: np.ndarray

    def __init__(self, matpath: str):
        classes = ["Healthy", "HGC", "IDC", "LGC"]
        modalities = ["1450_bgsub", "1668_bgsub", "SHG"]

        images_list: list[np.ndarray] = []
        labels_list: list[int] = []
        ids_list: list[str] = []

        # Verify files exist
        for classname in classes:
            mat_filename = f"{classname}_bulk_data.mat"
            if not os.path.isfile(os.path.join(matpath, mat_filename)):
                raise FileNotFoundError(
                    f"Expected file '{mat_filename}' not found in path '{matpath}'"
                )

            names_filename = f"{classname}_names.txt"
            if not os.path.isfile(os.path.join(matpath, names_filename)):
                raise FileNotFoundError(
                    f"Expected file '{names_filename}' not found in path '{matpath}'"
                )

        for classname in classes:
            mat_filename = f"{classname}_bulk_data.mat"
            names_filename = f"{classname}_names.txt"

            raw_data: dict[str, Any] = cast(
                dict[str, Any], loadmat(os.path.join(matpath, mat_filename))
            )

            with open(
                os.path.join(matpath, names_filename), "r", encoding="utf-8"
            ) as f:
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

                    assert len(names) == int(
                        mode_data.shape[2]
                    ), f"Number of names in '{names_filename}' does not match number of samples in '{mat_filename}' for modality '{mode}'"

                    mode_data_images.append(mode_data)  # (width, height, n samples)

                multimodal_data = np.stack(
                    mode_data_images, axis=-1
                )  # (width, height, n samples, n modalities)
                multimodal_data = np.transpose(
                    multimodal_data, (2, 3, 0, 1)
                )  # (n samples, n modalities, width, height)

                images_list.append(multimodal_data)
                labels_list.extend([classes.index(classname)] * len(names))
                ids_list.extend([name[0] for name in names])

        self.images = np.concatenate(images_list, axis=0)
        self.class_labels = np.array(labels_list)
        self.patient_ids = np.array(ids_list)

    def get_dims(self) -> tuple[int, int, int, int]:
        """
        Returns the dimensions of the image data as (n samples, n modalities, width, height).
        """
        return self.images.shape
