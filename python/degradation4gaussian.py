"""Yeonwoo: reproducible degradation. Float64 computation, clip to 0..255.
Returned floats are evaluated before PNG quantization; noise_sigma != filter_sigma.
"""
import numpy as np
from my_filters import conv2d, gaussian_kernel


def gaussian_noise(img, noise_sigma=20, seed=0):
    if not np.isfinite(noise_sigma) or noise_sigma < 0: raise ValueError('invalid noise sigma')
    x = np.asarray(img, dtype=float)
    return np.clip(x + np.random.default_rng(seed).normal(0, noise_sigma, x.shape), 0, 255)


def salt_pepper(img, amount=0.05, seed=0):
    if not 0 <= amount <= 1: raise ValueError('invalid amount')
    x = np.array(img, dtype=float, copy=True)
    u = np.random.default_rng(seed).random(x.shape)
    x[u < amount/2] = 0
    x[(u >= amount/2) & (u < amount)] = 255
    return x


def gaussian_blur(img, blur_sigma=2):
    return np.clip(conv2d(img, gaussian_kernel(blur_sigma)), 0, 255)

