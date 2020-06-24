import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
import scipy.signal
import scipy.ndimage
# x = [
# 	[[0, 1, 2, 3, 4],
# 		[5, 6, 7, 8, 9],
# 		[10, 11, 12, 13, 14],
# 		[15, 16, 17, 18, 19],
# 		[20, 21, 22, 23, 24]],
# 	[[0, 1, 2, 3, 4],
# 		[5, 6, 7, 8, 9],
# 		[10, 11, 12, 13, 14],
# 		[15, 16, 17, 18, 19],
# 		[20, 21, 22, 23, 24]],
# 	[[0, 1, 2, 3, 4],
# 		[5, 6, 7, 8, 9],
# 		[10, 11, 12, 13, 14],
# 		[15, 16, 17, 18, 19],
# 		[20, 21, 22, 23, 24]]
# ]
#
# detector = [
# 	[[0, 1, 0],
# 		[0, 1, 0],
# 		[0, 1, 0]],
# 	[[0, 1, 0],
# 		[0, 1, 0],
# 		[0, 1, 0]],
# 	[[0, 1, 0],
# 		[0, 1, 0],
# 		[0, 1, 0]]
# ]
#
# x = np.array(x)
# detector = np.array(detector)
#
#
# pad = (3 - 1) // 2
# x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), mode='constant')
# stride = 1
#
# o_dims = int(np.floor(((5 + (2 * pad) - 1 * (3 - 1) - 1) / stride) + 1))
#
# sub_dims = int(np.floor(((5 + (2 * pad) - 1 * (3 - 1) - 1) / 1) + 1))
# sub_shape = (1, sub_dims, sub_dims)
# # sub_shape = tuple(np.subtract(x.shape, x.shape) + 1)
#
# submatrices_ = np.lib.stride_tricks.as_strided(x, detector.shape + sub_shape, (392, 56, 8, 392, 56, 8))
# convolved_matrix = np.einsum('xyz,xyzklm->klm', detector, submatrices_)
#
# if stride == 2:
# 	stride_shape = (convolved_matrix.strides[0], convolved_matrix.strides[1] * 2, convolved_matrix.strides[2] * 2)
# 	convolved_matrix = np.lib.stride_tricks.as_strided(convolved_matrix, (1, o_dims, o_dims), stride_shape)
#
# print(convolved_matrix)

A = np.array([[[[-0.1010, -0.0570], [-0.0215,  0.5785]]]])

stride = 2
output = np.zeros((A.shape[0], A.shape[1], A.shape[2] * stride, A.shape[3] * stride))

for img in range(A.shape[0]):
	for f in range(A.shape[1]):
		output[img, f, ...] = scipy.ndimage.zoom(A[img, f, ...], stride, order=0)

print(output)
