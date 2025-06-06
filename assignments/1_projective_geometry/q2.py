import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

import utils


np.set_printoptions(precision=4)

# General settings and original image.
feature: str = "painting"
img_path: str = f"data/q1/{feature}.jpg"
par_annotations_path: str = f"data/annotation/q1_{feature}.npy"
per_annotations_path: str = f"data/annotation/q2_{feature}.npy"

img: np.ndarray = cv.imread(img_path, cv.IMREAD_COLOR_RGB)

# Read the perpendicular line annotations.
per_points: np.ndarray = utils.read_annotations(
    per_annotations_path,
    feature,
)[:8]

# Read the parallel annotations.
par_points: np.ndarray = utils.read_annotations(par_annotations_path, feature)

# Affine rectification.
affine_h: np.ndarray = utils.affine_rectification(par_points)
print("Affine rectification matrix:\n", affine_h)

# Computing affine-rectified image and annotated points.
aff_img: np.ndarray = utils.warp(img, affine_h)
aff_points: np.ndarray = utils.normalize(per_points @ affine_h.T)

# Metric rectification.
metric_h: np.ndarray = utils.metric_rectification(aff_points)
print("Metric rectification matrix:\n", metric_h)

# Computing metric-rectified image and annotated points.
met_img: np.ndarray = utils.warp(aff_img, metric_h)
met_points: np.ndarray = utils.normalize(aff_points @ metric_h.T)

# Testing perpendicularity.
met_h_inv: np.ndarray = np.linalg.inv(metric_h)
aff_h_inv: np.ndarray = np.linalg.inv(affine_h)

cosines: list[float] = []
for j in range(0, len(per_points), 4):
    l1: np.ndarray = utils.cross(per_points[j], per_points[j + 1])
    l2: np.ndarray = utils.cross(per_points[j + 2], per_points[j + 3])
    l1a = aff_h_inv.T @ l1
    l2a = aff_h_inv.T @ l2
    l1_prime: np.ndarray = met_h_inv.T @ l1a
    l2_prime: np.ndarray = met_h_inv.T @ l2a
    cosines.append((utils.cosine(l1, l2), utils.cosine(l1_prime, l2_prime)))

print("Cosines before and after metric rectification:")
print(np.array(cosines))

# Image generation.
fix, axes = plt.subplots(1, 4, figsize=(15, 5))

for axis in axes:
    axis.axis("off")

axes[0].imshow(img)
axes[0].set_title("Original Image")


def draw_annotated_img(
    img: np.ndarray,
    points: np.ndarray,
    axis: plt.Axes,
    title: str,
) -> None:
    axis.imshow(img)
    axis.plot(points[0:2, 0], points[0:2, 1], c="#0F0")
    axis.plot(points[2:4, 0], points[2:4, 1], c="#0F0")
    axis.plot(points[4:6, 0], points[4:6, 1], c="#00F")
    axis.plot(points[6:8, 0], points[6:8, 1], c="#00F")
    axis.set_title(title)


# Draw images.
draw_annotated_img(img, per_points, axes[1], "Annotated Original Image")
draw_annotated_img(aff_img, aff_points, axes[2], "Affine-Rectified Image")
draw_annotated_img(met_img, met_points, axes[3], "Metric-Rectified Image")

plt.tight_layout()
plt.savefig(f"submission/output/q2_{feature}.jpg")
