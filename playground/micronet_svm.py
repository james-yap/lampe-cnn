import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pretrained_microscopy_models as pmm
import torch.utils.model_zoo as model_zoo

# Assume MatReader is imported
from shared.mat_reader import MatReader
from shared.constants import CLASS_NAMES


class NLOMicroscopyDataset(Dataset):
    """
    A PyTorch Dataset that wraps the MatReader data.
    Handles tensor conversion, channel-wise standardization, and integer encoding
    of patient/core IDs for PyTorch DataLoader compatibility.
    """

    def __init__(self, mat_reader: MatReader):
        super().__init__()

        # 1. Convert NumPy arrays to PyTorch tensors
        # shape: (N, C, H, W) -> float32 for network weights
        self.images = torch.from_numpy(mat_reader.images).float()
        self.labels = torch.from_numpy(mat_reader.class_labels).long()

        # 2. Integer Encode Core IDs
        # PyTorch dataloaders work best with numeric types. We map the string
        # patient IDs (e.g., '1_B1') to unique integer group IDs.
        unique_ids = np.unique(mat_reader.patient_ids)
        id_to_int: dict[str, int] = {pid: i for i, pid in enumerate(unique_ids)}

        # Create a tensor of these integer group IDs
        group_ints = [id_to_int[pid] for pid in mat_reader.patient_ids]
        self.groups = torch.tensor(group_ints, dtype=torch.long)

        # 3. Channel-wise Standardization (Crucial for multi-modal NLO data)
        # We normalize each channel (SHG, SRS 1450, SRS 1668) independently across
        # the entire dataset to have a mean of 0 and std of 1.
        num_channels = self.images.shape[1]
        for c in range(num_channels):
            mean = self.images[:, c, :, :].mean()
            std = self.images[:, c, :, :].std()

            # Prevent division by zero if a channel is completely blank
            if std > 1e-8:
                self.images[:, c, :, :] = (self.images[:, c, :, :] - mean) / std
            else:
                self.images[:, c, :, :] = self.images[:, c, :, :] - mean

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the normalized image tensor, the class label, and the core ID integer."""
        return self.images[idx], self.labels[idx], self.groups[idx]


def extract_micronet_features(
    dataloader: DataLoader, model: nn.Module, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs the dataset through the frozen model to extract embeddings.
    Returns standard NumPy arrays ready for scikit-learn.
    """
    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_groups: list[np.ndarray] = []

    # Disable gradient tracking to dramatically reduce memory and speed up extraction
    with torch.no_grad():
        for images, labels, groups in dataloader:
            images = images.to(device)

            # The model output is the 2048-dim embedding because we replaced the fc layer
            features = model(images)

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
            all_groups.append(groups.numpy())

    # Stack the lists into flat 1D and 2D arrays
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    groups = np.concatenate(all_groups)

    return X, y, groups


def run_hybrid_pipeline(mat_data_path: str) -> None:
    """
    Main orchestration function:
    Loads data, extracts features via MicroNet, and trains the SVM.
    """
    print("Loading data via MatReader...")
    mat_reader = MatReader(mat_data_path)
    dataset = NLOMicroscopyDataset(mat_reader)

    # Batch size can be relatively large since we aren't storing gradients
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Setup Device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "mps"
    print(f"Using device: {device}")

    # Initialize MicroNet Feature Extractor
    print("Initializing MicroNet ResNet50...")
    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=False)
    url = pmm.util.get_pretrained_microscopynet_url("resnet50", "micronet")
    model.load_state_dict(model_zoo.load_url(url, map_location=device))

    # Strip the classification head to expose the 2048-dimensional pooling layer
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()  # Freeze batchnorm/dropout

    # Extract deep features
    print("Extracting features (this may take a moment)...")
    X, y, groups = extract_micronet_features(dataloader, model, device)
    print(f"Feature extraction complete. Feature matrix shape: {X.shape}")

    # Setup SVM with Grouped K-Fold
    # Using 'poly' kernel and 'ovo' (one-vs-one) to mirror the ECOC approach.
    gkf = GroupKFold(n_splits=5)
    svm_classifier = SVC(
        kernel="poly", class_weight="balanced", decision_function_shape="ovo"
    )

    fold_accuracies: list[float] = []
    all_y_true: list[int] = []
    all_y_pred: list[int] = []

    print("\nStarting Grouped Cross-Validation...")
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale SVM inputs. Fit ONLY on training data to avoid data leakage.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train and Predict
        svm_classifier.fit(X_train_scaled, y_train)
        y_pred = svm_classifier.predict(X_test_scaled)

        # Record Metrics
        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(float(acc))
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        print(f"Fold {fold+1} Accuracy: {acc * 100:.2f}%")

    print("\n--- Final Results ---")
    print(
        f"Mean CV Accuracy: {np.mean(fold_accuracies) * 100:.2f}% ± {np.std(fold_accuracies) * 100:.2f}%"
    )

    target_names = CLASS_NAMES  # Adjust based on your CLASS_NAMES constant
    print("\nGlobal Classification Report:")
    print(classification_report(all_y_true, all_y_pred, target_names=target_names))


if __name__ == "__main__":
    # Point this to the directory containing your _bulk_data.mat files
    mat_directory_path = "lampe_dataset/Full images"
    run_hybrid_pipeline(mat_directory_path)
