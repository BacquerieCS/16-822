import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

import utils


np.set_printoptions(precision=4)

# General settings and original image.
feature: str = "ceiling"
img_path: str = f"data/q1/{feature}.jpg"
annotations_path: str = f"data/annotation/q1_{feature}.npy"
output_path: str = f"submission/output/q1_{feature}.jpg"

img: np.ndarray = cv.imread(img_path, cv.IMREAD_COLOR_RGB)

# Read the annotations.
points = utils.read_annotations(annotations_path, feature)

# Constructing the homography.
affine_h: np.ndarray = utils.affine_rectification(points)
print("Affine rectification matrix:\n", affine_h)

# Testing parallelism.
h_inv: np.ndarray = np.linalg.inv(affine_h)

cosines: list[float] = []
for i in range(0, len(points), 4):
    l1: np.ndarray = utils.cross(points[i], points[i + 1])
    l2: np.ndarray = utils.cross(points[i + 2], points[i + 3])
    l1_prime: np.ndarray = l1.T @ h_inv
    l2_prime: np.ndarray = l2.T @ h_inv
    cosines.append((utils.cosine(l1, l2), utils.cosine(l1_prime, l2_prime)))

print("Cosines before and after affine rectification:")
print(np.array(cosines))

# Compute affine-rectified image and annotated points.
aff_image: np.ndarray = utils.warp(img, affine_h)
aff_points: np.ndarray = utils.normalize(points @ affine_h.T)

# Image generation.
fix, axes = plt.subplots(1, 3, figsize=(15, 5))

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
draw_annotated_img(img, points, axes[1], "Annotated Original Image")
draw_annotated_img(aff_image, aff_points, axes[2], "Affine-Rectified Image")
plt.tight_layout()
plt.savefig(output_path)
