import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import utils


def metric_rectification_orthogonal(line_pairs):
    """
    Computes metric rectification homography from orthogonal line pairs.

    Parameters
    ----------
    line_pairs : list of tuples
        Each tuple contains two orthogonal lines (homogeneous coordinates).
        Must contain at least 5 line pairs.

    Returns
    -------
    H_metric : ndarray (3x3)
        Metric rectification homography matrix.
    """
    # Build linear equations: each orthogonal pair gives 1 equation
    A = []
    for l, m in line_pairs:
        a1, b1, c1 = l
        a2, b2, c2 = m
        A.append([
            a1*a2,
            a1*b2 + b1*a2,
            b1*b2,
            a1*c2 + c1*a2,
            b1*c2 + c1*b2,
            c1*c2
        ])
    A = np.array(A)

    # Solve system via SVD
    _, _, Vt = np.linalg.svd(A)
    s = Vt[-1]

    # Form the dual conic matrix from s
    C_star_inf = np.array([
        [s[0], s[1]/2, s[3]/2],
        [s[1]/2, s[2], s[4]/2],
        [s[3]/2, s[4]/2, s[5]]
    ])

    # Enforce rank-2 constraint explicitly (important!)
    U, D, Vt = np.linalg.svd(C_star_inf)
    D[2] = 0  # enforce rank-2 explicitly
    C_star_inf_rank2 = U @ np.diag(D) @ Vt

    # Extract affine-to-metric rectification homography:
    # upper-left 2x2 submatrix decomposition
    # (the ideal conic is [[I,0],[0,0]], thus extract only upper-left block)
    C_affine = C_star_inf_rank2[:2, :2]

    # Check definiteness for stability
    eigvals, eigvecs = np.linalg.eigh(C_affine)

    sqrt_D = np.diag(np.sqrt(eigvals))
    K = eigvecs @ sqrt_D @ eigvecs.T

    # Metric rectification homography (upper-left 2x2 block inverse of K)
    H_metric = np.eye(3)
    H_metric[:2, :2] = np.linalg.inv(K)

    return H_metric


# General settings.
feature: str = "tiles5"
img_path: str = f"data/q1/{feature}.jpg"
annotations_path: str = f"data/annotation/q2_annotation.npy"
output_path: str = f"output/q4_{feature}.jpg"

# Reading the image and annotations.
img: np.ndarray = cv.imread(f"data/q1/{feature}.jpg", cv.IMREAD_COLOR_RGB)
points: np.ndarray = utils.read_annotations(annotations_path, feature)


all_lines = utils.get_lines(points)
lines = []
for i in range(0, len(all_lines), 2):
    lines.append((all_lines[i], all_lines[i+1]))

metric_h = metric_rectification_orthogonal(lines)
print("H", metric_h, metric_h.dtype)
metric_img: np.ndarray = cv.warpPerspective(img, metric_h, (img.shape[1], img.shape[0]))

# Testing perpendicularity.
met_h_inv: np.ndarray = np.linalg.inv(metric_h)

cosines: list[float] = []
for j in range(0, len(points), 4):
    l1: np.ndarray = utils.cross(points[j], points[j + 1])
    l2: np.ndarray = utils.cross(points[j + 2], points[j + 3])
    l1_prime: np.ndarray = met_h_inv.T @ l1
    l2_prime: np.ndarray = met_h_inv.T @ l2
    cosines.append((utils.cosine(l1, l2), utils.cosine(l1_prime, l2_prime)))

print("Cosines before and after metric rectification:")
print(np.array(cosines))

plt.imshow(metric_img)
plt.savefig(output_path)
