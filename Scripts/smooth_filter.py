#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
Copyright (C) 2024  Andreas Langer

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
============================================================================    
   
Created on Sat Jun  8 12:16:17 2024

GNU GENERAL PUBLIC LICENSE Version 3

@author: Andreas Langer
"""

import numpy as np
import tensorflow as tf

def smooth_filter(A, blur_para, sm_p):
    """
    Computes the convolution of a given n1xn2 matrix A with 
    a Gaussian filter (sm_p == 'g') or a mean filter (sm_p == 'm').
    The filter-window-size is determined by blur_para.
    """
    # Make sure the filter size is odd
    m = blur_para[0] + (1 - blur_para[0] % 2)

    n3 = A.shape[1]

    if n3 == 1:
        print('A is not an image!')
        return A

    rm = (m - 1) // 2
    
    if sm_p == 'g':
        # Gaussian filter
        sigma = blur_para[1]
        H = gaussian_kernel(m, sigma)
        
        A = convolve_with_padding(A, H, rm)
    elif sm_p == 'm':
        # Mean filter
        H = np.ones((m, m)) / (m * m)
        A = convolve_with_padding(A, H, rm)

    return A

def gaussian_kernel(size, sigma):
    """Generate a 2D Gaussian kernel."""
    k = size // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    normalizing_factor = 1 / (2 * np.pi * sigma**2)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel*normalizing_factor
    kernel /= kernel.sum()
    return kernel

def convolve_with_padding(A, H, pad_x, pad_y=None):
    """Convolves A with H, including padding."""
    if pad_y is None:
        pad_y = pad_x
    
    size = H.shape[0]
    
    H_tf = tf.convert_to_tensor(H, dtype=tf.float32)
    H_tf = tf.reshape(H_tf, [size, size, 1, 1])
    A_tf = tf.convert_to_tensor(A, dtype=tf.float32)
    N1,N2 = A_tf.shape
    
    A_tf = tf.reshape(A_tf, [1, N1, N2, 1])  # Reshape to [batch, height, width, channels]
    A_padded = tf.pad(A_tf, paddings=[[0, 0], [pad_y, pad_y], [pad_x, pad_x], [0, 0]], mode='SYMMETRIC')
    output_tf = tf.nn.conv2d(A_padded, H_tf, strides=[1, 1, 1, 1], padding='VALID')
    output_squeezed_tf = tf.squeeze(output_tf)
    return output_squeezed_tf


