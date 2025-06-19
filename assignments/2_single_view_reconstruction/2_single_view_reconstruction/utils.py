import cv2 as cv
import numpy as np


def camera_matrix(points_2d: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """
    Computes the camera matrix P given a set of homogeneous correspondences.
    """
    # Construction of the matrix A, in ``Ap = 0``.
    a: list = []
    for (xp, yp, _), (x, y, z, _) in zip(points_2d, points_3d):
        a.append([-x, -y, -z, -1, 0, 0, 0, 0, x * xp, y * xp, z * xp, xp])
        a.append([0, 0, 0, 0, -x, -y, -z, -1, x * yp, y * yp, z * yp, yp])
    # Null space computation.
    _, _, p = np.linalg.svd(a)
    p = (p[-1] / p[-1][-1])
    # Camera matrix P.
    return p.reshape(3, 4)


def draw_lines(
    img: np.ndarray,
    points: np.ndarray,
    w: float = 3,
) -> np.ndarray:
    """
    Draws lines on an image, where each line is constructed from 2 consecutive
    `points` (going like 0-1, 2-3, ...).

    :param w: Line width.
    """
    new_img: np.ndarray = np.copy(img)
    for i in range(0, len(points), 2):
        p1: np.ndarray = points[i].astype(int)
        p2: np.ndarray = points[i + 1].astype(int)
        new_img = cv.line(new_img, p1, p2, color=(0, 255, 0), thickness=w)
    return new_img


def draw_points(img: np.ndarray, points: np.ndarray, w: int = 3) -> np.ndarray:
    """
    Draws a set of inhomogeneous 2D `points` in `img`.

    :param w: Point width.
    """
    new_img: np.ndarray = img.copy()
    for x, y in points:
        x, y = int(x), int(y)
        new_img[y - w: y + w, x - w: x + w] = 0, 255, 0
    return new_img


def read_annotations(annotations_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads the point annotations in `annotation_path`, assuming it contains
    several rows, each with 5 numbers: (2 for 2D points, 3 for 3D points).

    :returns: A tuple of 2 numpy arrays: ``(2D points, 3D points)``, each in
      homogeneous coordinates.
    """
    points_2d, points_3d = [], []
    with open(annotations_path) as file:
        for line in file:
            points: list[float] = [float(x) for x in line.split(" ")]
            points_2d.append(points[:2] + [1.0])
            points_3d.append(points[2:] + [1.0])
    return np.array(points_2d), np.array(points_3d)
