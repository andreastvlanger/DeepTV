#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
Copyright (C) 2024  Andreas Langer, Sara Behnamian

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

Created on Wed Jul  3 09:56:55 2024

GNU GENERAL PUBLIC LICENSE Version 3

@authors: Andreas Langer, Sara Behnamian
"""

import numpy as np
from skimage.util import random_noise
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim
import scipy.sparse as sp

import skimage
from packaging import version

def add_gauss_noise(org, gnoise, seed_nr, c='real'):
    np.random.seed(seed_nr)  # Set the seed for reproducibility
    std_dev = gnoise[1] # Standard deviation
    expected_value = gnoise[0]
    if c == 'real':
        b = expected_value + std_dev * np.random.randn(*org.shape)
    elif c == 'complex':
        # Generate real and imaginary components separately
        xs = np.random.randn(*org.shape)
        ys = np.random.randn(*org.shape)
        b = std_dev / np.sqrt(2) * (xs + 1j * ys)

    g = org + b
    return g


def add_salt_pepper_noise(g, noise_level=0, sp_ratio=0.5, seed_nr=0):
    np.random.seed(seed_nr)
    # noise_level .... Proportion of image pixels to replace with noise
    # sp_ratio .... Proportion of salt vs. pepper noise. Higher values represent more salt
    
    current_version = skimage.__version__
    target_version = "0.21.0"
    if version.parse(current_version) < version.parse(target_version):
        g = random_noise(g, mode='s&p', seed=seed_nr, amount=noise_level, salt_vs_pepper=sp_ratio)
    else:
        g = random_noise(g, mode='s&p', rng=seed_nr, amount=noise_level, salt_vs_pepper=sp_ratio)
    return g


 
def add_random_impulse_noise(g, rvnoise, seed_nr=0):
    """
    Adds random-valued impulse noise with probability rvnoise to an image g.
    
    Parameters:
    g (numpy.ndarray): Input image as a 2D matrix.
    rvnoise (float): Probability of the random impulse noise.
    
    Returns:
    numpy.ndarray: Image with added random-valued impulse noise.
    """
    N1, N2 = g.shape
    if min(N1, N2) == 1:
        # Handle special case where the image is effectively 1D
        pass
    else:
        G = g.flatten()
        a = np.min(G)
        b = np.max(G)
        
        np.random.seed(seed_nr)  # Seed the random number generator for reproducibility
        Coordinates = np.random.permutation(len(G))
        nrCoordinates = round(rvnoise * len(Coordinates))
        
        np.random.seed(seed_nr)  # Seed again for reproducibility
        values = a + (b - a) * np.random.rand(nrCoordinates)
        nCoord = Coordinates[:nrCoordinates]
        
        for i in range(nrCoordinates):
            G[nCoord[i]] = values[i]
        
        g = G.reshape(N1, N2)
    
    return g

def input_image(image):
    path = './Images/'
    # Define image filenames and types
    image_files = {
        9: path + 'LU1.png'
    }
    
    # Check if the image_index is None or not in the dictionary
    if image is None or image not in image_files:
        raise ValueError("No valid image selected. Please provide a valid image index.")

    print('Load image', image_files[image])
    
    orgim = image_files.get(image, 'default_image.jpg')
    try:
        org = imread(orgim)
        # print(org.shape)
        if len(org.shape) == 3 and org.shape[2] == 3:
            org = rgb2gray(org)  # Convert to grayscale if it's RGB
        elif len(org.shape) == 3 and org.shape[2] == 4:
            org = rgb2gray(org[..., :3])
            
        org = img_as_float(org)  # Convert image to float
    except Exception as e:
        raise IOError(f"Image file not found. Please check the image path. {e}")
 
    if np.max(org) <=1:
        if np.min(org)>=0:
            print('Image is in range [0,1].')
            return org 
    # Scale the image to [0,1] according to different versions
    version = 3
    if version == 1:  # Scale such that max value is 1
        org = org / np.max(org)
    elif version == 2:  # Scale to range [0, 1]
        org = (org - np.min(org)) / (np.max(org) - np.min(org))
    elif version == 3:  # Assume values are in range 0-255 and normalize to [0,1]
        org = org / 255
        print('Image is assumed to be in range [0,255] and scaled to [0,1]!')
    print(np.max(org), np.min(org))
    return org

def calculate_quality_measures(org, g):
    """
    Calculate quality measures between original and reconstructed images.
    
    Parameters:
        org (numpy.ndarray): Original image.
        g (numpy.ndarray): Reconstructed image.
    
    Returns:
        tuple: PSNR, SSIM, MAE
    """
    N1, N2 = org.shape

    # PSNR calculation
    mse = np.mean((org - g) ** 2)
    PSNR = 20 * np.log10(1 / np.sqrt(mse))

    # Convert to 8-bit images assuming inputs are in [0, 1]
    U = (g * 255).astype(np.uint8)
    G = (org * 255).astype(np.uint8)

    # SSIM calculation
    SSIM = ssim(G, U, data_range=G.max() - G.min())

    # MAE calculation
    MAE = np.mean(np.abs(org - g))

    return PSNR, SSIM, MAE

def function_nabla(m1, m2, hx, hy, bc='Neumann'):
    if bc=='Neumann': #homogeneous Neumann boundary conditions
        if m2==1:
            m1,m2=m2,m1
            
        m = m1 * m2

        ##################################################################
        E = np.ones((m1,m2))
        E3 = np.copy(E)
        E3[:,-1] = 0
        
        e3a = np.reshape(E3,(m,))
        
        one_zero = np.array([0]).reshape(1,)
        e3b = np.concatenate([one_zero, e3a[0:m-1]])
        
        data_DX = np.vstack((-e3a,e3b))
        offset_DX=np.array([0,1])
        DX = sp.spdiags(data_DX, offset_DX, m,m)
        
        #####################################################
        E4a=np.copy(E)
        E4b=np.copy(E)
        E4a[-1,:]=0
        E4b[0,:]=0
        e4a=np.reshape(E4a,(m,))
        e4b=np.reshape(E4b,(m,))
        
        data_DY = np.vstack((-e4a,e4b))
        offset_DY=np.array([0,m1])
        
        DY = sp.spdiags(data_DY, offset_DY, m, m)
        GRAD=sp.vstack([(1/hx)*DX, (1/hy)*DY])
        
        if m1==1:
            GRAD = (1/hx)*DX
        elif m2==1:
            print('You should really transpose your vector!')#???
        return GRAD
    elif bc=='Dirichlet': #homogeneous Dirichlet boundary conditions
        if m2==1:
            m1,m2=m2,m1
            
        m = m1*m2
        E = np.ones((m1,m2))
        e = np.reshape(E, (m,))
        
        E1 = np.copy(E)
        E1[:,-1] = 0
        e1a = np.reshape(E1,(m,))
        one_zero = np.array([0]).reshape(1,)
        e1b = np.concatenate([one_zero, e1a[0:m-1]])
        data_DX = np.vstack((-e,e1b))
        offset_DX=np.array([0,1])
        DX = sp.spdiags(data_DX, offset_DX, m,m)
        
        data_DY = np.vstack((-e,e))
        offset_DY=np.array([0,m1])
        DY = sp.spdiags(data_DY, offset_DY, m,m)
        
        if m1 ==1:
            GRAD = (1/hx)*DX
        else:
            GRAD=sp.vstack([(1/hx)*DX, (1/hy)*DY])
        
        return GRAD
    
def check_no_further_dimension(arr):
    """
    Check if there is no further dimension beyond the first one.
    
    Parameters:
    arr (numpy.ndarray): Input array.

    Returns:
    bool: True if there is no further dimension, False otherwise.
    """
    try:
        _ = arr.shape[1]
        return False  # Shape[1] exists, meaning there is another dimension
    except IndexError:
        return True  # Shape[1] does not exist, meaning it's a 1D array
    

def function_eta(v,m, tv_type='TV2'):
    # Input: vector v of length d*m
    # Output: l2-norm of v of length m
    # Reshape and split the input vector
    
    if check_no_further_dimension(v)==1:
        m1=v.shape[0]
        m2=1
    else:
        m1,m2 = v.shape
        
    if m1 == m or m2 == m:
        eta_r = np.abs(v)
    else:
        if tv_type=='TV21':
            eta_r1 = np.hstack((v[:m],v[m:2*m]))
            eta_r2 = np.hstack((v[2*m:3*m],v[3*m:]))
        else:
            eta_r = np.hstack((v[:m], v[m:2*m]))
      
        # Compute the sum of squares along rows and take the square root
        if tv_type=='TV2':
            eta_r = np.reshape(np.sqrt(np.sum(np.abs(eta_r) ** 2, axis=1)),(m,1)) #TV_2 isotropic
        elif tv_type == 'TV21':
            eta_r1 = np.reshape(np.sqrt(np.sum(np.abs(eta_r1) ** 2, axis=1)),(m,1)) #l2-norm
            eta_r2 = np.reshape(np.sqrt(np.sum(np.abs(eta_r2) ** 2, axis=1)),(m,1)) #l2-norm
            eta_r = 0.5*(eta_r1+eta_r2)
    return eta_r

def ForwardDiffX(m1, m2, hx, bc='Neumann'):
    if bc=='Neumann': #homogeneous Neumann boundary conditions
        if m2==1:
            m1,m2=m2,m1
            
        m = m1 * m2

        ##################################################################
        E = np.ones((m1,m2))
        E3 = np.copy(E)
        E3[:,-1] = 0
        
        e3a = np.reshape(E3,(m,))
        
        one_zero = np.array([0]).reshape(1,)
        e3b = np.concatenate([one_zero, e3a[0:m-1]])
        
        data_DX = np.vstack((-e3a,e3b))
        offset_DX=np.array([0,1])
        DX = sp.spdiags(data_DX, offset_DX, m,m)
        return (1/hx)*DX
    elif bc=='Dirichlet': #homogeneous Dirichlet boundary conditions
        if m2==1:
            print('You should transpose your vector!')
            m1,m2=m2,m1
            
        m = m1*m2
        E = np.ones((m1,m2))
        e = np.reshape(E, (m,))
        
        E1 = np.copy(E)
        E1[:,-1] = 0
        e1a = np.reshape(E1,(m,))
        one_zero = np.array([0]).reshape(1,)
        e1b = np.concatenate([one_zero, e1a[0:m-1]])
        data_DX = np.vstack((-e,e1b))
        offset_DX=np.array([0,1])
        DX = sp.spdiags(data_DX, offset_DX, m,m)
        return (1/hx)*DX
    
def ForwardDiffY(m1, m2, hy, bc='Neumann'):
    if bc=='Neumann': #homogeneous Neumann boundary conditions
        if m2==1:
            m1,m2=m2,m1
            
        m = m1 * m2
        E = np.ones((m1,m2))
        E4a=np.copy(E)
        E4b=np.copy(E)
        E4a[-1,:]=0
        E4b[0,:]=0
        e4a=np.reshape(E4a,(m,))
        e4b=np.reshape(E4b,(m,))
        
        data_DY = np.vstack((-e4a,e4b))
        offset_DY=np.array([0,m1])
        
        DY = sp.spdiags(data_DY, offset_DY, m, m)
        return (1/hy)*DY
    
    elif bc=='Dirichlet': #homogeneous Dirichlet boundary conditions
        if m2==1:
            print('You should transpose your vector!')
            m1,m2=m2,m1
            
        m = m1*m2
        E = np.ones((m1,m2))
        e = np.reshape(E, (m,))
        
        data_DY = np.vstack((-e,e))
        offset_DY=np.array([0,m1])
        DY = sp.spdiags(data_DY, offset_DY, m,m)
        return (1/hy)*DY
 
def BackwardDiffX(m1, m2, hx, bc='Neumann'): 
    if bc=='Neumann': #homogeneous Neumann boundary conditions
        if m2==1:
            m1,m2=m2,m1
            
        m = m1 * m2
        
       
        ##################################################################
        E = np.ones((m1,m2))
        E3 = np.copy(E)
        E3[:,-1] = 0
        
        e3a = np.reshape(E3,(m,))
        
        one_zero = np.array([0]).reshape(1,)
        e3b = np.concatenate([one_zero, e3a[0:m-1]])
        
        data_DX = np.vstack((-e3a,e3b))
        offset_DX=np.array([-1,0])
        DX = sp.spdiags(data_DX, offset_DX, m,m)
        return (1/hx)*DX
    elif bc=='Dirichlet': #homogeneous Dirichlet boundary conditions
        if m2==1:
            print('You should transpose your vector!')
            m1,m2=m2,m1
            
        m = m1*m2
        E = np.ones((m1,m2))
        e = np.reshape(E, (m,))
        
        E1 = np.copy(E)
        e1a = np.reshape(E1,(m,))
        one_zero = np.array([0]).reshape(1,)
        data_DX = np.vstack((-e,e1a))
        offset_DX=np.array([-1,0])
        DX = sp.spdiags(data_DX, offset_DX, m,m)
        return (1/hx)*DX
    
def BackwardDiffY(m1, m2, hy, bc='Neumann'):
    if bc=='Neumann': #homogeneous Neumann boundary conditions
        if m2==1:
            m1,m2=m2,m1
            
        m = m1 * m2
        E = np.ones((m1,m2))
        E4a=np.copy(E)
        E4b=np.copy(E)
        E4a[-1,:]=0
        E4b[0,:]=0
        e4a=np.reshape(E4a,(m,))
        e4b=np.reshape(E4b,(m,))
        
        data_DY = np.vstack((-e4a,e4b))
        offset_DY=np.array([-m1,0])
        
        DY = sp.spdiags(data_DY, offset_DY, m, m)
        return (1/hy)*DY
    
    elif bc=='Dirichlet': #homogeneous Dirichlet boundary conditions
        if m2==1:
            print('You should transpose your vector!')
            m1,m2=m2,m1
            
        m = m1*m2
        E = np.ones((m1,m2))
        e = np.reshape(E, (m,))

        
        data_DY = np.vstack((-e,e))
        offset_DY=np.array([-m1,0])
        DY = sp.spdiags(data_DY, offset_DY, m,m)
        return (1/hy)*DY   
    
def function_nabla_FB(m1, m2, hx, hy, bc='Neumann'): #gradient for forward-backward differences!
        DX1=ForwardDiffX(m1, m2, hx, bc=bc)
        DY1=ForwardDiffY(m1, m2, hy, bc=bc)
        DX2=BackwardDiffX(m1, m2, hx, bc=bc)
        DY2=BackwardDiffY(m1, m2, hy, bc=bc)
        
        GRAD = sp.vstack([DX1, DY1, DX2, DY2])
        return GRAD
