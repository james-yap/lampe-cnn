"""
MatReader class implementation.
"""

import os
from typing import Any, cast
import numpy as np
from scipy.io import loadmat  # type: ignore

from shared.constants import CLASS_NAMES, MODALITIES


class MatReader:
    """
    Takes in a directory path and extracts .mat files by class.
    Unified interface for loading LAMPE datasets into PyTorch DataLoaders.
    Automatically groups data by patient to prevent intracore bias.
    """

    # shape: np.ndarray(n samples, n modalities, height, width)
    images: np.ndarray

    # shape: np.ndarray(n samples,)
    # 0: HGC, 1: IDC
    class_labels: np.ndarray

    # Known duplicate FOVs in the stable dataset: indices 19/38 were originally
    # labelled HGC, while the same FOV IDs also appear as IDC at indices 103/112.
    # Treat the duplicated FOVs as IDC (class 1) post-hoc.
    _FORCE_CLASS_1_INDICES = np.array([19, 38], dtype=np.int64)

    # shape: np.ndarray(n samples,)
    patient_ids: np.ndarray

    # shape: np.ndarray(n samples,)
    fov_ids: np.ndarray

    def __init__(self, matpath: str):

        images_list: list[np.ndarray] = []
        labels_list: list[int] = []
        ids_list: list[str] = []
        fov_ids_list: list[str] = []

        # Verify files exist
        for classname in CLASS_NAMES:
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

        for classname in CLASS_NAMES:
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

                for mode in MODALITIES:
                    mode_key = f"{classname}_{mode}"
                    mode_data = np.asarray(cast(np.ndarray, raw_data[mode_key]))

                    if mode_data.ndim < 3:
                        raise ValueError(
                            f"Expected 3D array for '{mode_key}', got shape {mode_data.shape}"
                        )

                    assert len(names) == int(mode_data.shape[2]), (
                        f"Number of names in '{names_filename}' does not match"
                        f"number of samples in '{mat_filename}' for modality '{mode}'"
                    )

                    mode_data_images.append(mode_data)  # (width, height, n samples)

                multimodal_data = np.stack(
                    mode_data_images, axis=-1
                )  # (height, width, n samples, n modalities) - MATLAB column-major ordering
                multimodal_data = np.transpose(
                    multimodal_data, (2, 3, 0, 1)
                )  # (n samples, n modalities, height, width)

                images_list.append(multimodal_data)
                labels_list.extend([CLASS_NAMES.index(classname)] * len(names))
                ids_list.extend([f"{name[0]}_{name[1]}" for name in names])
                fov_ids_list.extend(
                    [f"{name[0]}_{name[1]}_{name[3]}" for name in names]
                )

                # if classname == "LGC":
                #     mirrored_data = np.flip(multimodal_data, axis=-1)
                #     images_list.append(mirrored_data)
                #     labels_list.extend([CLASS_NAMES.index(classname)] * len(names))
                #     ids_list.extend([f"{name[0]}_{name[1]}" for name in names])

        self.images = np.concatenate(images_list, axis=0)
        self.class_labels = np.array(labels_list)
        self.patient_ids = np.array(ids_list)
        self.fov_ids = np.array(fov_ids_list)

        if self.get_num_fovs() > int(self._FORCE_CLASS_1_INDICES.max()):
            self.class_labels[self._FORCE_CLASS_1_INDICES] = 1

        # Geometric augmentation: reflection across vertical axis (flip left-right)
        # flipped = np.flip(self.images, axis=-1)
        # self.images = np.concatenate([self.images, flipped], axis=0)
        # self.class_labels = np.tile(self.class_labels, 2)
        # self.patient_ids = np.tile(self.patient_ids, 2)

        # Geometric augmentation: reflection across horizontal axis (flip up-down)
        # flipped_ud = np.flip(self.images, axis=-2)
        # self.images = np.concatenate([self.images, flipped_ud], axis=0)
        # self.class_labels = np.tile(self.class_labels, 2)
        # self.patient_ids = np.tile(self.patient_ids, 2)

        # Geometric augmentation: add 90°, 180°, 270° rotations (axes 2,3 = H,W)
        # rot90 = np.rot90(self.images, k=1, axes=(2, 3))
        # rot180 = np.rot90(self.images, k=2, axes=(2, 3))
        # rot270 = np.rot90(self.images, k=3, axes=(2, 3))
        # self.images = np.concatenate([self.images, rot90, rot180, rot270], axis=0)
        # self.class_labels = np.tile(self.class_labels, 4)
        # self.patient_ids = np.tile(self.patient_ids, 4)

    def get_dims(self) -> tuple[int, int, int, int]:
        """
        Returns the dimensions of the image data as (n samples, n modalities, height, width).
        """
        return self.images.shape

    def get_num_fovs(self) -> int:
        """
        Returns the number of FOVs (samples) in the dataset.
        """
        return self.images.shape[0]

    def get_num_channels(self) -> int:
        """
        Returns the number of modalities (channels) in the dataset.
        """
        return self.images.shape[1]

    def get_height_width(self) -> tuple[int, int]:
        """
        Returns the height and width of the images in the dataset.
        """
        return self.images.shape[2], self.images.shape[3]
