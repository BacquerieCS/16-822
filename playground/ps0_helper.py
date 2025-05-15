import math
import numpy as np

# 1 VECTOR SPACES
print("=" * 50)
print("1. VECTOR SPACES")

# QUESTION 3
a: np.ndarray = np.array([[-3, -9, 1, 1], [-3, -9, 1, 2], [-3, -9, 1, 3]])
x1: np.ndarray = np.array([[-3], [1], [0], [0]])
x2: np.ndarray = np.array([[-2], [0], [-6], [0]])

print("QUESTION 3")
print(a)
print(a @ x1)
print(a @ x2)

# 2 EIGENVALUES, EIGENVECTOR AND SVD
print("=" * 50)
print("2. EIGENVALUES, EIGENVECTOR AND SVD")

# QUESTION 1
print("QUESTION 1")

a: np.ndarray = np.array([[1, 2], [2, 4]])
print(a)
print(np.linalg.eig(a))

# QUESTION 4
print("QUESTION 4")

c: float = 4353245
a: np.ndarray = np.array([[1, 0], [c, 1]])

disc: float = c * math.sqrt(c**2 + 4)
eig1: float = math.sqrt((c**2 + 2 + disc) / 2)
eig2: float = math.sqrt((c**2 + 2 - disc) / 2)

print(eig1, eig2)
print(np.sqrt(np.linalg.eigvals(a @ a.T)))
