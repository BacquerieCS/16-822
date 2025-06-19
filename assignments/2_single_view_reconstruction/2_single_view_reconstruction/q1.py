"""
Implementation required for "Q1: Camera matrix P from 2D-3D correspondences".
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

import utils


# Configuration and initialization.
img_path: str = "data/q1/cube/cube.jpg"
annotations_path: str = "data/q1/cube/edges.txt"
#surface_points_path: str = "data/q1/bunny_pts.npy"
bounding_box_path: str = "data/q1/cube/cube_pts.npy"

img: np.ndarray = cv.imread(img_path, cv.IMREAD_COLOR_RGB)

# Surface points.
# surface_points_3d: np.ndarray = np.load(surface_points_path)
# ones: np.ndarray = np.ones((len(surface_points_3d), 1))
# surface_points_3d = np.concatenate((surface_points_3d, ones), axis=1)

# Bounding box points.
bounding_box_3d: np.ndarray = np.load(bounding_box_path).reshape(-1, 3)
ones: np.ndarray = np.ones((len(bounding_box_3d), 1))
bounding_box_3d = np.concatenate((bounding_box_3d, ones), axis=1)

points_2d, points_3d = utils.read_annotations(annotations_path)

# Camera matrix computation.
p: np.ndarray = utils.camera_matrix(points_2d, points_3d)
print("Camera matrix P:\n", p)

# 3D -> 2D surface point mapping.
# surface_points_2d: np.ndarray = surface_points_3d @ p.T
# surface_points_2d /= surface_points_2d[:, -1][:, None]

# 3D -> 2D bounding box mapping.
bounding_box_2d: np.ndarray = bounding_box_3d @ p.T
bounding_box_2d /= bounding_box_2d[:, -1][:, None]

# Outputting results.
fig, ax = plt.subplots(1, 3, figsize=(25, 15))

[a.axis("off") for a in ax]

# Original image.
ax[0].set_title("Original Image")
ax[0].imshow(img)

# Annotated 2D points.
recovered_2d: np.ndarray = points_3d @ p.T
recovered_2d /= recovered_2d[:, -1][:, None]
ax[1].set_title("Annotated 2D Points")
ax[1].imshow(utils.draw_points(img, recovered_2d[:, :2], 5))

# Surface points.
# ax[1, 0].set_title("Surface Points")
# ax[1, 0].imshow(utils.draw_points(img, surface_points_2d[:, :2], 2))

# Bounding box.
ax[2].set_title("Point Mapping")
ax[2].imshow(utils.draw_lines(img, bounding_box_2d[:, :2], 4))

plt.savefig("output/q1_cube.png")
