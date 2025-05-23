import numpy as np


# 1. 2D PROJECTIVE GEOMETRY
print("=" * 50)
print("1. 2D PROJECTIVE GEOMETRY")

# QUESTION 1
print("-> 1a.")
p1: np.ndarray = np.array([3, 4, 1])
p2: np.ndarray = np.array([4, 3, 0])
print(np.cross(p1, p2))

print("-> 1b.")
p1: np.ndarray = np.array([3, 4, 2022])
p2: np.ndarray = np.array([3, 4, -1967])
print(np.cross(p1, p2))

print("-> 2a.")
l1: np.ndarray = np.array([3, 4, 1])
l2: np.ndarray = np.array([0, 0, 1])
print(np.cross(l1, l2))

print("-> 2b.")
l1: np.ndarray = np.array([3, 4, 1])
l2: np.ndarray = np.array([3, 4, 2])
print(np.cross(l1, l2))

print("-> 3a.")
h: np.ndarray = np.array([[3, 0, 0], [0, 2, 0], [0, 0, 1]])
h_1: np.ndarray = np.linalg.inv(h)
p: np.ndarray = np.array([[3, 4, 1]]).T
print(h @ p)

print("-> 3b.")
l: np.ndarray = np.array([[-4, 3, 0]]).T
print(h_1 @ l)

print("-> 3c.")
c: np.ndarray = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
print(h_1.T @ c @ h_1)
