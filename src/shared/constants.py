"""
Constants shared across multiple modules.
"""

import torch

# CLASS_NAMES = ["Healthy", "LGC", "HGC", "IDC"]
CLASS_NAMES = ["HGC", "IDC"]

MODALITIES = ["SHG", "1450_bgsub", "1668_bgsub"]
# MODALITIES = ["1450_onres", "1668_onres", "SHG"]


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _select_device()
