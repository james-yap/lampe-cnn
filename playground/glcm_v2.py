import numpy as np
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score
from collections import Counter

# Import provided classes
from shared.mat_reader import MatReader
from shared.constants import CLASS_NAMES

# Aggression order for tie-breaking: IDC-P > HGC > LGC > Benign
# Assuming CLASS_NAMES is ordered ['Benign', 'LGC', 'HGC', 'IDC-P']
# If not, map them to an aggressiveness score explicitly.
AGGRESSION_HIERARCHY = {
    name: i for i, name in enumerate(["Benign", "LGC", "HGC", "IDC-P"])
}


def normalize_to_128_gray_levels(image: np.ndarray) -> np.ndarray:
    """
    Min-max normalizes an image to 128 gray levels.
    The top 1% of pixels are capped to prevent bright outliers from skewing the distribution.
    """
    p99 = np.percentile(image, 99)
    img_clipped = np.clip(image, a_min=None, a_max=p99)

    img_min = img_clipped.min()
    img_max = img_clipped.max()

    if img_max == img_min:
        return np.zeros_like(img_clipped, dtype=np.uint8)

    img_norm = ((img_clipped - img_min) / (img_max - img_min)) * 127
    return img_norm.astype(np.uint8)


def extract_features(subimage: np.ndarray) -> np.ndarray:
    """
    Extracts 4 first-order and 5 second-order statistics from a normalized subimage.
    """
    # 1. First-order statistics
    flat_img = subimage.flatten()
    mean_val = np.mean(flat_img)
    var_val = np.var(flat_img)
    skew_val = skew(flat_img)
    kurt_val = kurtosis(flat_img)

    # 2. Second-order statistics (GLCM)
    # Offsets: Horizontal (0), Diagonal-Up (pi/4), Vertical (pi/2), Diagonal-Down (3pi/4)
    distances = [1]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    # Calculate symmetric, normalized GLCM
    glcm = graycomatrix(
        subimage,
        distances=distances,
        angles=angles,
        levels=128,
        symmetric=True,
        normed=True,
    )

    # Calculate standard props provided by skimage
    contrast = np.mean(graycoprops(glcm, "contrast"))
    correlation = np.mean(graycoprops(glcm, "correlation"))
    energy = np.mean(graycoprops(glcm, "energy"))
    homogeneity = np.mean(graycoprops(glcm, "homogeneity"))

    # Calculate Entropy manually (-sum(P * ln(P))) for each angle, then average
    entropy_vals = []
    for a in range(len(angles)):
        P = glcm[:, :, 0, a]
        P_nz = P[P > 0]  # Avoid log(0)
        entropy = -np.sum(P_nz * np.log(P_nz))
        entropy_vals.append(entropy)
    entropy_avg = np.mean(entropy_vals)

    return np.array(
        [
            mean_val,
            var_val,
            skew_val,
            kurt_val,
            contrast,
            correlation,
            energy,
            homogeneity,
            entropy_avg,
        ]
    )


def prepare_dataset(mat_reader: MatReader, channel_idx: int):
    """
    Splits FOVs into 3x3 grids and extracts features.
    channel_idx: 0 for SHG, 1 for SRS 1450, 2 for SRS 1668 (based on MODALITIES indexing).
    """
    X_features = []
    y_labels = []
    groups = []
    fov_indices = []

    num_fovs = mat_reader.get_num_fovs()

    for fov_idx in range(num_fovs):
        image = mat_reader.images[fov_idx, channel_idx, :, :]
        label = mat_reader.class_labels[fov_idx]
        patient_id = mat_reader.patient_ids[fov_idx]

        norm_img = normalize_to_128_gray_levels(image)
        h, w = norm_img.shape

        # Grid step sizes
        h_step, w_step = h // 3, w // 3

        # Split into 3x3 subimages
        for i in range(3):
            for j in range(3):
                subimage = norm_img[
                    i * h_step : (i + 1) * h_step, j * w_step : (j + 1) * w_step
                ]

                features = extract_features(subimage)

                X_features.append(features)
                y_labels.append(label)
                groups.append(patient_id)
                fov_indices.append(
                    fov_idx
                )  # Keep track of parent FOV for majority voting

    return (
        np.array(X_features),
        np.array(y_labels),
        np.array(groups),
        np.array(fov_indices),
    )


def fov_majority_vote(
    subimage_preds: np.ndarray, subimage_fov_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregates subimage predictions back to the FOV level using mode.
    Resolves ties using the aggression hierarchy.
    """
    unique_fovs = np.unique(subimage_fov_indices)
    fov_preds = []

    for fov in unique_fovs:
        preds_in_fov = subimage_preds[subimage_fov_indices == fov]
        counts = Counter(preds_in_fov)

        # Find the maximum count
        max_count = max(counts.values())
        candidates = [cls for cls, count in counts.items() if count == max_count]

        if len(candidates) == 1:
            fov_preds.append(candidates[0])
        else:
            # Tie breaker: Pick the most aggressive class among the tied candidates
            candidate_names = [CLASS_NAMES[c] for c in candidates]
            most_aggressive_name = max(
                candidate_names, key=lambda x: AGGRESSION_HIERARCHY.get(x, -1)
            )
            fov_preds.append(CLASS_NAMES.index(most_aggressive_name))

    return unique_fovs, np.array(fov_preds)


def run_experiment(matpath: str, channel_idx: int):
    print("Loading data...")
    mat_reader = MatReader(matpath)

    print("Extracting features from subimages...")
    X, y, groups, fov_indices = prepare_dataset(mat_reader, channel_idx)

    # SVM Setup: Polynomial kernel, balanced weights, One-vs-One (ECOC)
    svm = OneVsOneClassifier(SVC(kernel="poly", class_weight="balanced"))

    # 2:1 Train/Test Split based on Groups (Patient Cores)
    gss = GroupShuffleSplit(n_splits=10, test_size=0.33, random_state=42)

    subimage_accuracies = []
    fov_accuracies = []

    print("Running 10-fold Monte Carlo cross-validation...")
    for fold, (train_idx, test_idx) in enumerate(gss.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_fov_indices = fov_indices[test_idx]

        # Train model
        svm.fit(X_train, y_train)

        # Subimage level evaluation
        sub_preds = svm.predict(X_test)
        sub_acc = accuracy_score(y_test, sub_preds)
        subimage_accuracies.append(sub_acc)

        # FOV level evaluation (Majority Consensus)
        unique_fovs, fov_preds = fov_majority_vote(sub_preds, test_fov_indices)

        # Get true labels for the unique FOVs
        # We can just take the first true label encountered for each FOV since all subimages share the FOV label
        fov_trues = []
        for fov in unique_fovs:
            idx = np.where(test_fov_indices == fov)[0][0]
            fov_trues.append(y_test[idx])

        fov_acc = accuracy_score(fov_trues, fov_preds)
        fov_accuracies.append(fov_acc)

        print(
            f"Iteration {fold+1}: Subimage Acc: {sub_acc:.4f} | FOV Acc: {fov_acc:.4f}"
        )

    print("\n--- Final Results (10 Iteration Average) ---")
    print(
        f"Mean Subimage Classification Accuracy: {np.mean(subimage_accuracies)*100:.2f}%"
    )
    print(f"Mean FOV Majority-Consensus Accuracy:  {np.mean(fov_accuracies)*100:.2f}%")


if __name__ == "__main__":
    # Assuming channel 0 is SHG, 1 is SRS 1450, 2 is SRS 1668
    TARGET_CHANNEL = 0
    DATA_PATH = "lampe_dataset/3x3 bad SHG removed"
    run_experiment(DATA_PATH, TARGET_CHANNEL)
