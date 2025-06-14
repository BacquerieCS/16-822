import numpy as np
import matplotlib.pyplot as plt


X: np.ndarray = np.array([4, 3, 2])
f: float = 1

z: float = X[-1].item()
x: np.ndarray = ((f / z) * X)[:2]

print(X)
print(x)


# 3D point
X = np.array([4, 3, 2])

fig = plt.figure(figsize=(10, 4))

# 3D subplot
ax3d = fig.add_subplot(1, 2, 1, projection='3d')
ax3d.scatter(*X, c='r', s=100)
ax3d.set_title('3D Point')
ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('Z')
ax3d.set_xlim([0, 5])
ax3d.set_ylim([0, 5])
ax3d.set_zlim([0, 5])

# Optionally, show the camera center at the origin
ax3d.scatter(0, 0, 0, c='k', marker='^', s=80, label='Camera')
ax3d.legend()

# Projected point
x_proj = np.array([2, 1.5])  # from previous computation

# 2D subplot
ax2d = fig.add_subplot(1, 2, 2)
ax2d.scatter(*x_proj, c='b', s=100)
ax2d.set_title('Projected 2D Image Point')
ax2d.set_xlabel('x')
ax2d.set_ylabel('y')
ax2d.set_xlim([0, 5])
ax2d.set_ylim([0, 5])
ax2d.grid(True)

plt.tight_layout()
plt.show()
