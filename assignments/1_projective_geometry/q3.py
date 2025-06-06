import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

import utils


np.set_printoptions(precision=4)

# General settings and original images.
normal_img_path: str = "data/q3/painting-normal.jpg"
persp_img_path: str = "data/q3/tv-perspective.jpg"
annotations_path: str = f"data/annotation/q3_tv-perspective.npy"

# Reading images.
normal_img: np.ndarray = cv.imread(normal_img_path, cv.IMREAD_COLOR_RGB)
perspective_img: np.ndarray = cv.imread(persp_img_path, cv.IMREAD_COLOR_RGB)

# Annotations.
h, w = normal_img.shape[:2]
normal_points: np.ndarray = utils.as_homogeneous(
    np.array([[0, 0], [w, 0], [w, h], [0, h]])
)
perspective_points: np.ndarray = utils.read_annotations(
    annotations_path,
    "tv-perspective",
)


def find_projection(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    a: list[list[float]] = []
    for (x, y), (xp, yp) in zip(src, tgt):
        a.append([x, y, 1, 0, 0, 0, -xp * x, -xp * y, -xp])
        a.append([0, 0, 0, x, y, 1, -yp * x, -yp * y, -yp])

    _, _, sol = np.linalg.svd(a)
    h: np.ndarray = sol[-1].reshape(3, 3)
    return h / h[2, 2]


homography: np.ndarray = find_projection(
    normal_points[:, :2],
    perspective_points[:, :2],
)
print("Homography:\n", homography)

ph, pw = perspective_img.shape[:2]
warped_img: np.ndarray = cv.warpPerspective(normal_img, homography, (pw, ph))

mask = np.any(warped_img != 0, -1)
output_img: np.ndarray = perspective_img.copy()
output_img[mask] = warped_img[mask]

fig, axes = plt.subplots(1, 3, figsize=(15, 10))

for axis in axes:
    axis.axis("off")

axes[0].imshow(normal_img)
axes[0].set_title("Original Image")
axes[1].imshow(perspective_img)
axes[1].set_title("Perspective Image")
axes[2].imshow(output_img)
axes[2].set_title("Warped and Overlaid Image")

plt.savefig("submission/output/q3_painting.jpg")
