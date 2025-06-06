import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.pyplot import Axes
from matplotlib.pyplot import Figure


def affine_rectification(points: np.ndarray) -> np.ndarray:
    """
    Computes the affine rectification matrix for the given homogeneous
    annotations `points`.
    """
    # Parallel lines.
    l1, l2, m1, m2 = get_lines(points)[:4]
    p1, p2 = [cross(l1, l2), cross(m1, m2)]
    # Perspective line at infinity.
    a, b, c = cross(p1, p2)
    return np.array([[1, 0, 0], [0, 1, 0], [a, b, c]])


def get_lines(points: np.ndarray) -> list[np.ndarray]:
    """
    Constructs the lines found in `points`, each as a row vector.
    """
    return [cross(points[i], points[i + 1]) for i in range(0, len(points), 2)]


def annotate(path: str) -> list[tuple[float, ...]]:
    """
    Returns the points clicked on the image at `path` as a list. Each
    (homogeneous) point has the form ``(x, y, 1.0)``.
    """
    clicks: list[tuple[float, ...]] = []

    img: np.ndarray = np.array(Image.open(path))

    fig: Figure = plt.figure(1)
    ax: Axes = fig.add_subplot(111)
    ax.imshow(img)

    def click(event):
        x, y = event.xdata, event.ydata
        plt.plot(x, y, "ro")
        clicks.append((x, y))

    _ = fig.canvas.mpl_connect("button_press_event", click)
    plt.show()

    return np.array(clicks)


def as_homogeneous(points: np.ndarray) -> np.ndarray:
    """
    Converts the points in `points` to homogeneous coordinates.

    :param points: A matrix of shape ``(n, 2)``.
    """
    n: int = len(points)
    return np.concatenate((points, np.ones((n, 1))), axis=1)


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    """
    Computes the cosine of the angle between vectors `u` and `v`.
    """
    u, v = u[:2], v[:2]
    norms: float = np.linalg.norm(u) * np.linalg.norm(v)
    return (np.dot(u, v) / norms).item()


def cross(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Computes the cross-product between the homogeneous vectors `u` and `v`,
    and returns a vector with a ``z`` component of 1.
    """
    return normalize(np.array([np.cross(u, v)]))[0]


def metric_rectification(points: np.ndarray) -> np.ndarray:
    """
    Computes the metric rectification matrix for the given homogeneous
    annotations `points`.
    """
    # Equation system construction.
    lines: np.ndarray = get_lines(points)
    a: np.ndarray = np.zeros((len(lines) // 2, 3), np.float64)
    j: int = 0
    for i in range(0, len(lines), 2):
        l1: np.ndarray = lines[i]
        l2: np.ndarray = lines[i + 1]
        a[j] = l1[0] * l2[0], l1[0] * l2[1] + l1[1] * l2[0], l1[1] * l2[1]
        j += 1
    # Reconstruction of the ideal circular points.
    _, _, sol = np.linalg.svd(a)
    s: np.ndarray = sol[-1]
    s /= s[-1]
    sym: np.ndarray = np.array([[s[0], s[1]], [s[1], s[2]]])
    k: np.ndarray = np.linalg.cholesky(sym)
    # Metric rectification matrix.
    metric_h: np.ndarray = np.eye(3)
    metric_h[:2, :2] = np.linalg.inv(k)
    return metric_h


def normalize(v: np.ndarray) -> np.ndarray:
    """
    Normalizes the homogenous vector `v`.
    """
    z: float = v[:, 2].reshape(-1, 1)
    return np.where(z == 0, v, v / z)


def read_annotations(file_path: str, feature: str) -> np.ndarray:
    """
    Reads the annotated points from `file_path` and returns them as
    homogeneous row vectors.
    """
    with open(file_path, "rb") as file:
        annotations = np.load(file, allow_pickle=True)
    return as_homogeneous(annotations.item().get(feature))


def warp(img: np.ndarray, proj: np.ndarray) -> np.ndarray:
    """
    Applies the perspective projection `proj` to the image `img`.
    """
    h, w = img.shape[:2]

    points: np.ndarray = np.array([[0, 0], [0, h], [w, h], [w, 0]], np.float64)
    points = points.reshape(-1, 1, 2)
    points = cv2.perspectiveTransform(points, proj)

    x_min, y_min = (points.min(axis=0).ravel() - 0.5).astype(int)
    x_max, y_max = (points.max(axis=0).ravel() + 0.5).astype(int)

    ht: np.ndarray = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    return cv2.warpPerspective(img, ht @ proj, (x_max - x_min, y_max - y_min))
