import numpy as np


# Configuration and initial data.
img_path: str = "data/q2b.png"
annotations_path: str = "data/q2/q2b.npy"

annotations: np.ndarray = np.load(annotations_path)
points: np.ndarray = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], np.float64)


# Homography computations.
def find_homography(points: np.ndarray, points_p: np.ndarray) -> np.ndarray:
    """
    Computes the planar homography that maps `points` to `points_p`, assuming
    them inhomogeneous.
    """
    # Constructs the matrix ``A`` for ``Ax = 0``.
    a: list = []
    for (x, y), (xp, yp) in zip(points, points_p):
        a.append([x, y, 1, 0, 0, 0, -x * xp, - y * xp, -xp])
        a.append([0, 0, 0, x, y, 1, -x * yp, - y * yp, -yp])
    a: np.ndarray = np.array(a)
    # Null space of ``A``.
    _, _, h = np.linalg.svd(a)
    h = (h[-1] / h[-1][-1])
    # Final homography.
    return h.reshape(3, 3)


homographies: np.ndarray = np.zeros((3, 3, 3), np.float64)

for i in range(3):
    homographies[i] = find_homography(points, annotations[i])


# Intrinsics K computation.
def find_intrinsics(homographies: np.ndarray) -> np.ndarray:
    """
    Computes the intrinsics matrix ``K`` given a set of `homographies`.
    """
    # Constructs the matrix ``A`` for ``Aw = 0``.
    a: list = []
    for h in homographies:
        h11, h12, _, h21, h22, _, h31, h32, _ = h.flatten()
        a.append(
            [
                h11 ** 2 + h12 ** 2,
                2 * (h11 * h21 + h12 * h22),
                2 * (h11 * h31 + h12 * h32),
                h21 ** 2 + h22 ** 2,
                2 * (h21 * h31 + h22 * h32),
                h31 ** 2 + h32 ** 2,
            ]
        )
    a: np.ndarray = np.array(a)
    # Null space of ``A``.
    _, _, w_raw = np.linalg.svd(a)
    w_raw = w_raw[-1] / w_raw[-1][-1]
    w: np.ndarray = np.array(
        [
            [w_raw[0], w_raw[1], w_raw[2]],
            [w_raw[1], w_raw[3], w_raw[4]],
            [w_raw[2], w_raw[4], w_raw[5]],
        ]
    )
    # Fallback to SVD
    u, s, vh = np.linalg.svd(w)
    K_inv = u @ np.diag(np.sqrt(s)) @ vh
    K = np.linalg.inv(K_inv)
    K /= K[-1, -1]
    return K


print(find_intrinsics(homographies))
