import numpy as np


src: np.ndarray = np.array(
    [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
    ]
)

tgt: np.ndarray = np.array(
    [
        [2, 1],
        [4, 1],
        [4, 2],
        [2, 3],
    ]
)


def find_projection(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    a: list[list[float]] = []
    for (x, y), (xp, yp) in zip(src, tgt):
        a.append([x, y, 1, 0, 0, 0, -xp * x, -xp * y, -xp])
        a.append([0, 0, 0, x, y, 1, -yp * x, -yp * y, -yp])

    _, _, sol = np.linalg.svd(a)
    h: np.ndarray = sol[-1].reshape(3, 3)
    return h / h[2, 2]


src_h: np.ndarray = np.concatenate((src, np.ones((src.shape[0], 1))), axis=1)
p = src_h @ find_projection(src, tgt).T
p = p / p[:, -1].reshape(-1, 1)
print(p)
