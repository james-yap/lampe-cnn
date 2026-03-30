import numpy as np
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings

from shared.mat_reader import MatReader

# Suppress warnings from division by zero in zero-variance patches
warnings.filterwarnings("ignore", category=RuntimeWarning)


def normalize_and_quantize(image: np.ndarray, num_levels: int = 128) -> np.ndarray:
    """
    Caps the top 1% of pixels, then min-max normalizes and quantizes to specified levels.
    """
    # Cap the top 1% of pixels to prevent aberrant bright pixels from skewing normalization
    p99 = np.percentile(image, 99)
    clipped_image = np.clip(image, a_min=None, a_max=p99)

    img_min = clipped_image.min()
    img_max = clipped_image.max()

    if img_max == img_min:
        return np.zeros_like(image, dtype=np.uint8)

    norm_img = (clipped_image - img_min) / (img_max - img_min)

    # Quantize to integer levels [0, num_levels - 1]
    quantized = np.floor(norm_img * (num_levels - 1)).astype(np.uint8)
    return quantized


def shannon_entropy(glcm: np.ndarray) -> np.ndarray:
    """
    Computes Shannon entropy from a GLCM matrix.
    glcm shape: (levels, levels, distances, angles)
    """
    entropy_vals = np.zeros((glcm.shape[2], glcm.shape[3]))
    for d in range(glcm.shape[2]):
        for a in range(glcm.shape[3]):
            # Extract the 2D matrix for this distance/angle
            p = glcm[:, :, d, a].astype(float)
            p_sum = np.sum(p)
            if p_sum > 0:
                # Normalize to probabilities
                p = p / p_sum
                # Compute entropy (ignoring p=0 to avoid log(0))
                entropy_vals[d, a] = -np.sum(p[p > 0] * np.log2(p[p > 0]))
    return entropy_vals


def extract_combined_features(images: np.ndarray) -> np.ndarray:
    """
    Extracts first and second-order statistics.
    Returns array of shape (n_samples, n_modalities * 9 features)
    """
    n_samples, n_modalities, h, w = images.shape
    num_levels = 128

    # Offsets: 1 pixel. Angles: 0, 45, 90, 135 degrees (horizontal, diag-up, vertical, diag-down)
    distances = [1]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    all_features = []

    print(f"Extracting first and second-order features for {n_samples} samples...")

    for i in range(n_samples):
        sample_features = []

        for m in range(n_modalities):
            img_channel = images[i, m, :, :]

            # --- First-Order Statistics ---
            # Calculated directly from pixel intensities
            flat_img = img_channel.flatten()
            mean_val = np.mean(flat_img)
            var_val = np.var(flat_img)
            skew_val = float(skew(flat_img))
            kurt_val = float(kurtosis(flat_img))

            # --- Second-Order Statistics (GLCM) ---
            quantized_img = normalize_and_quantize(img_channel, num_levels=num_levels)
            glcm = graycomatrix(
                quantized_img,
                distances=distances,
                angles=angles,
                levels=num_levels,
                symmetric=True,
                normed=True,
            )

            # Calculate properties and average across the 4 angles
            contrast = np.mean(graycoprops(glcm, "contrast"))
            homogeneity = np.mean(graycoprops(glcm, "homogeneity"))
            energy = np.mean(graycoprops(glcm, "energy"))
            correlation = np.mean(graycoprops(glcm, "correlation"))
            entropy = np.mean(shannon_entropy(glcm))

            # Combine all 9 features for this modality
            modality_features = [
                mean_val,
                var_val,
                skew_val,
                kurt_val,
                contrast,
                homogeneity,
                energy,
                correlation,
                entropy,
            ]
            sample_features.extend(modality_features)

        all_features.append(sample_features)

    return np.array(all_features, dtype=np.float32)


def run_pipeline(matpath: str):
    """
    Executes the pipeline using a 10-iteration 2:1 core-based data split.
    """
    reader = MatReader(matpath)
    X_raw = reader.images
    y = reader.class_labels
    groups = reader.patient_ids

    X_features = extract_combined_features(X_raw)

    # 10 iterations of a 2:1 (train:test) split grouped by patient core
    n_splits = 10
    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=0.33, random_state=42)

    accuracies = []
    cumulative_cm = np.zeros((4, 4), dtype=int)

    scaler = StandardScaler()
    # Polynomial kernel as specified by the paper's ECOC design
    svm = SVC(kernel="poly", degree=3, C=1.0, class_weight="balanced", random_state=42)

    print(f"Training and testing polynomial SVM over {n_splits} iterations...")

    for i, (train_idx, test_idx) in enumerate(splitter.split(X_features, y, groups)):
        X_train, X_test = X_features[train_idx], X_features[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        svm.fit(X_train_scaled, y_train)
        y_pred = svm.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        cumulative_cm += confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])

    print(
        f"\nMean Classification Accuracy across 10 folds: {np.mean(accuracies):.2%} ± {np.std(accuracies):.2%}"
    )
    print("\nCumulative Confusion Matrix (Healthy, LGC, HGC, IDC):")
    print(cumulative_cm)


run_pipeline("lampe_dataset/3x3 bad SHG removed")
