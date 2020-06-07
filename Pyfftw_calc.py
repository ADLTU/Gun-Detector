import pyfftw
from numpy.fft  import fft2, ifft2
import numpy as np
from Image import *


class PwfftwConv:
    def __init__(self, image, weights, o_dim, num_filters, biases, threads=4):

        self.image_obj = pyfftw.builders.rfftn(image, s=image.shape, threads=threads)
        self.filter_obj = pyfftw.builders.rfftn(weights, s=image.shape, threads=threads)
        self.ifft_obj = pyfftw.builders.irfftn(self.image_obj.output_array, s=(o_dim, o_dim, image.shape[2]-1), threads=threads)

    def __call__(self, image, weights, o_dim, num_filters, biases):
        self.bias = biases
        return self.ifft_obj((self.image_obj(image) * self.filter_obj(weights))).sum(axis=2) # + self.bias


class PwfftwConvd:
    def __init__(self, image, weights, o_dim, num_filters):
        self.image_obj = pyfftw.builders.rfftn(image, s=image.shape, threads=4)
        self.filter_obj = pyfftw.builders.rfftn(weights, s=image.shape, threads=4)
        self.ifft_obj = pyfftw.builders.irfftn(self.image_obj.output_array, s=(o_dim, o_dim,image.shape[2]-1), threads=4)

    def np_fftconvolve(self, A, B):
        try:
            return np.real(self.ifft_obj((self.image_obj(A) * self.image_obj(B))).sum(axis=2))
        except ValueError as e:
            print(e)
            exit()
        except IndexError as e:
            print(e)
            exit()

    def test_numpy_fft(self, A, B, o_dim, num_filters, biases):
        C = np.zeros((o_dim, o_dim, num_filters))
        for i_N in np.arange(B.shape[0]):
            D = B[i_N, :, :, :]
            C[:, :, i_N] = self.np_fftconvolve(A[:, :, :], D[:, :, :])
        return C