import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import oaconvolve

np.random.seed(0)

sigmas = np.array([1, 0.5, 0.7, 1.5])
n_sigma = sigmas.size
n_points = 50

point = np.zeros((4, 4))
l = 50
ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
xx, yy = np.meshgrid(ax, ax)

kernel = (np.square(xx) + np.square(yy)) / 1

kernel = np.exp(-0.5 * kernel)
kernel /= kernel.sum()
plt.figure()
plt.imshow(kernel)


# Generating 2D kernel with depth

sigmas = np.random.rand(n_points)

# xxz = np.expand_dims(xx, axis=2)
xx = np.tile(xx, (n_points, 1, 1))

# yy = np.expand_dims(yy, axis=2)
yy = np.tile(yy, (n_points, 1, 1)) 

kernel = (np.square(xx) + np.square(yy)) / np.square(sigmas.reshape(n_points, 1, 1))

kernel = np.exp(-0.5 * kernel)
kernel /= kernel.max()
print("kernel shape", kernel.shape)
plt.figure()
plt.title("ker")
plt.imshow(kernel[3, :])

image = np.zeros((n_points, 50,50))
points = np.random.randint(0, 50, size=(2, n_points))
points = np.vstack([points, np.arange(n_points).reshape(1, n_points)])

print(image.shape)
print(points.shape)
print(points[2], points[0], points[1], "end")
image[points[2], points[0], points[1]] = 1

conv_image = oaconvolve(image, kernel, mode="same", axes=(1, 2))

print(conv_image.shape)

plt.figure()
plt.title("im")
plt.imshow(conv_image.sum(axis=0))
plt.colorbar()
plt.show()