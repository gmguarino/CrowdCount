import numpy as np
import scipy
from sklearn.neighbors import KDTree
from scipy.signal import oaconvolve, convolve2d
import scipy
from time import time
# import torch 
# import torch.nn as nn
# import matplotlib.pyplot as plt

# class GaussianLayer(nn.Module):
#     def __init__(self, size, sigmas):
#         super(GaussianLayer, self).__init__()
#         self.depth = len(sigmas)
#         self.size = size
#         self.sigmas = sigmas
#         self.filter = nn.Conv2d(
#             self.depth,
#             self.depth, 
#             kernel_size=size,  
#             padding=size // 2, 
#             bias=False
#         )

#         self.weights_init()

#     def forward(self, x):
#         return self.filter(x)

#     def weights_init(self):
#         ax = np.linspace(-(self.size - 1) / 2., (self.size - 1) / 2., self.size)
#         xx, yy = np.meshgrid(ax, ax)
#         # Generating 2D kernel with depth

#         # xxz = np.expand_dims(xx, axis=2)
#         xx = np.tile(xx, (self.depth, 1, 1))

#         # yy = np.expand_dims(yy, axis=2)
#         yy = np.tile(yy, (self.depth, 1, 1))

#         kernel = (np.square(xx) + np.square(yy)) / np.square(self.sigmas.reshape(self.depth, 1, 1))

#         kernel = np.exp(-0.5 * kernel)
#         kernel /= 2 * np.pi * np.square(self.sigmas.reshape(self.depth, 1, 1))
#         self.filter.weight = nn.Parameter(torch.tensor(kernel, dtype=torch.float32).unsqueeze(0))
#         self.filter.weight.requires_grad_ = False



def volume_kernel(kernel_size, sigmas):
    n_points = sigmas.size
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    # Generating 2D kernel with depth

    # xxz = np.expand_dims(xx, axis=2)
    xx = np.tile(xx, (n_points, 1, 1))

    # yy = np.expand_dims(yy, axis=2)
    yy = np.tile(yy, (n_points, 1, 1)) 

    kernel = (np.square(xx) + np.square(yy)) / np.square(sigmas.reshape(n_points, 1, 1))

    kernel = np.exp(-0.5 * kernel)
    kernel /= 2 * np.pi * np.square(sigmas.reshape(n_points, 1, 1))

    return kernel

def format_index_arrays(gt, shape):
    index0 = gt[1, :]
    index0[index0 >= shape[0]] = shape[0] - 1
    index1 = gt[0, :]
    index1[index1 >= shape[1]] = shape[1] - 1
    return index0, index1

# def custom_convolve(image, kernel):
#     im_depth, im_height, im_width = image.shape
#     ker_depth, ker_height, ker_width = kernel.shape
#     convolution = np.zeros_like(image)
#     cube = np.zeros_like(kernel)
#     for j in range(im_height):
#         for i in range(im_width):
#             index0 = (j - ker_height / 2 >= 0 and j - ker_height / 2 >= 0)*(j - ker_height / 2)
#             index1 = min(i - ker_width / 2, 0)



def generate_density_map(ground_truth, shape):
    density_map = np.zeros((len(ground_truth), *shape))
    truth = ground_truth.copy()
    leafsize = 2048
    # build kdtree
    tree = KDTree(truth.copy(), leaf_size=leafsize)
    # query kdtree
    distances, locations = tree.query(truth, k=4)
    sigmas = np.mean(distances[:, 1:], axis=1) * 0.3

    # kernel = volume_kernel(min(*shape) // 20, sigmas)

    truth = truth.T
    truth = np.vstack([truth.astype('int'), np.arange(len(sigmas)).reshape(1, len(sigmas))])
    
    index0, index1 = format_index_arrays(truth, shape)
    density_map[truth[2, :], index0, index1] = 1
    for layer in range(density_map.shape[0]):

        density_map[layer, ...] = scipy.ndimage.filters.gaussian_filter(
            density_map[layer, ...],
            sigmas[layer],
            mode='constant'
        )
    print("convolution finished")
    # density_map = oaconvolve(density_map, kernel, mode="same", axes=(1, 2)) 

    return density_map.sum(axis=0)

# def generate_density_map_pytorch(ground_truth, shape):
#     density_map = np.zeros((len(ground_truth), *shape))
#     truth = ground_truth.copy()
#     leafsize = 2048
#     # build kdtree
#     tree = KDTree(truth.copy(), leaf_size=leafsize)
#     # query kdtree
#     distances, locations = tree.query(truth, k=4)
#     sigmas = np.mean(distances[:, 1:], axis=1) * 0.3

#     # kernel = volume_kernel(min(*shape) // 20, sigmas)
#     kernel = GaussianLayer(min(*shape) // 20, sigmas)

#     if torch.cuda.is_available():
#         kernel.cuda()

#     truth = truth.T
#     truth = np.vstack([truth.astype('int'), np.arange(len(sigmas)).reshape(1, len(sigmas))])
    
#     index0, index1 = format_index_arrays(truth, shape)
#     density_map[truth[2, :], index0, index1] = 1
#     # density_map = oaconvolve(density_map, kernel, mode="same", axes=(1, 2)) 
#     with torch.no_grad():
#         if torch.cuda.is_available():
#             density_map = kernel(torch.tensor(density_map, dtype=torch.float32).unsqueeze(0).cuda())
#         else:
#             density_map = kernel(torch.tensor(density_map, dtype=torch.float32).unsqueeze(0))

#     return density_map.sum(axis=0).cpu()

    


# kernel = GaussianLayer(15, np.random.randn(10)*10)
# kernel.requires_grad_ = False
# img = np.random.randn(1, 10, 500, 500)
# with torch.no_grad():
#     print(kernel(torch.tensor(img, dtype=torch.float32)).shape)
#     plt.imshow(kernel.filter.weight.numpy()[0,5,...])
# plt.show()