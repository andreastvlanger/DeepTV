#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:42:56 2024

MIT License

@authors: Andreas Langer, Sara Behnamian
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
import shutil
import traceback
import pickle
import NN_image_processing as imgNN

# Function to convert a list of tensors to a list of numpy arrays
def convert_tensors_to_numpy(tensors):
    return [tensor.numpy() for tensor in tensors]

# Function to set up logging by redirecting stdout to a log file
def setup_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'output.log')
    sys.stdout = open(log_file, 'w')


def save_essential_data(log_dir, **kwargs):
    file_path = os.path.join(log_dir, 'essential_data.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(kwargs, f)
    print(f"Saved essential data to {file_path}")
    print(f"Saved variables: {', '.join(kwargs.keys())}")    

# Define a piecewise constant function
def pc_function(x, value=1.0):
    return (np.where(x > 1, value, 0.0).astype(np.float32))

# TV-solution with g=pc_function
def pc_s_function(x, a, b, alpha1, alpha2, value=1.0):
    '''
    Parameters
    ----------
    x : np.array
        Input array.
    a : float
        Length of the left interval.
    b : float
        Length of the right interval.
    alpha1 : float
        Parameter alpha1.
    alpha2 : float
        Parameter alpha2.
    value : float, optional
        Value for the right piece of the function. The default is 1.0.

    Returns
    -------
    np.array
        Output array with piecewise linear values.
    '''
    return (np.where(x > 1, value - (1 - alpha1 * b) / (2 * alpha2 * b), 
                     (1 - alpha1 * a) / (2 * alpha2 * a)).astype(np.float32))

# Define energy functions
E = lambda a, b, alpha1, alpha2, value=1: (
    alpha1 * np.abs((1 - alpha1 * a) / (2 * alpha2 * a)) * a +
    alpha1 * np.abs(value - (value - (1 - alpha1 * b) / (2 * alpha2 * b))) * b +
    alpha2 * np.square((1 - alpha1 * a) / (2 * alpha2 * a)) * a +
    alpha2 * np.square(value - (value - (1 - alpha1 * b) / (2 * alpha2 * b))) * b +
    np.abs(value - (1 - alpha1 * b) / (2 * alpha2 * b) - (1 - alpha1 * b) / (2 * alpha2 * b))
)

E1 = lambda a, b, alpha1, alpha2, value=1: (
    alpha1 * np.abs((1 - alpha1 * a) / (2 * alpha2 * a)) * a +
    alpha1 * np.abs((1 - alpha1 * b) / (2 * alpha2 * b)) * b
)

E2 = lambda a, b, alpha1, alpha2, value=1: (
    alpha2 * np.square((1 - alpha1 * a) / (2 * alpha2 * a)) * a +
    alpha2 * np.square((1 - alpha1 * b) / (2 * alpha2 * b)) * b
)

ETV = lambda a, b, alpha1, alpha2, value=1: (
    np.abs(value - (1 - alpha1 * b) / (2 * alpha2 * b) - (1 - alpha1 * b) / (2 * alpha2 * b))
)

energy = lambda c, alpha1, alpha2: (
    alpha1 * np.abs(c[0]) + alpha1 * np.abs(1 - c[1]) +
    alpha2 * np.square(c[0]) + alpha2 * np.square(1 - c[1]) +
    np.abs(c[1] - c[0])
)

# Initialize parameters
d = 1  # Dimension (1D example)
interval = [0, 2]  # Interval for the domain
length = interval[1] - interval[0]  # Length of the interval
value = 1.0  # Value used in the piecewise function
n = int(1e3)  # Number of samples (n=1000)
h = length / n  # Step size for the grid

# Generate coordinates
x0 = np.linspace(interval[0], interval[1], n).reshape(-1, 1).astype(np.float32)

# Sample the original function at the coordinates
org = pc_function(x0, value=value)

# Define the problem type
problem = 'denoising'

# Set up the transformation T based on the problem type
if problem == 'denoising':
    parameter = ()  # No additional parameters needed for denoising
    T = imgNN.T_denoising  # Transformation function for denoising
elif problem == 'inpainting':
    mask = np.ones(org.shape).astype(np.float32)  # Create a mask for inpainting
    D = x0[int(n / 2) - 100:int(n / 2) + 100]  # Define the region to inpaint
    mask[int(n / 2) - 100:int(n / 2) + 100] = 0  # Set the mask to zero in the region
    parameter = (mask)  # Inpainting requires a mask as a parameter
    T = imgNN.T_inpainting(org, parameter)  # Transformation function for inpainting
elif problem == 'deblurring':
    pass  # TODO: Implement deblurring

# Add noise to the original signal
# noise added
# todo: Different noise-types and seed!
mu = 0  # Mean of the noise
sigma = 0  # Standard deviation of the noise
np.random.seed(0)  # Seed for reproducibility
noise = np.random.normal(mu, sigma, n).astype(np.float32)  # Generate noise
g = T(org, parameter) + noise.reshape((n, 1))  # Add noise to the transformed signal

# Plot the original and observed signals
plt.plot(x0, pc_function(x0, value=value), label='Original signal')
plt.plot(x0, g, label='Observed signal', linestyle='--')
if problem == 'inpainting':
    plt.plot(D, mask[int(n / 2) - 100:int(n / 2) + 100], linewidth=5)
plt.legend()
plt.show()

# Parameters for the model
alpha1 = 0.5  # Weight for L1 loss
alpha2 = 1.25  # Weight for L2 loss
alphaTV = 1.0  # Weight for total variation loss
rule = 'AR'  # Rule for discretization
epoch_nr = 100001  # Number of epochs for training

learning_rate = 0.01  # Learning rate for the optimizer

# Initialize the params dictionary
params = {
    'd': d,
    'h': h,
    'alpha1': alpha1,
    'alpha2': alpha2,
    'alpha_TV': alphaTV,
    'epochs': epoch_nr,
    'plot_interval': 100,
    'log_dir': None,  # This will be set later
    'rule': rule,
    'tv_type': 'TV2', 
    'boundary_condition': 'Neumann',
    'tv2_smoothing': 'Huber', # is not considered in 1d
    'gamma': 1e-10,
    'learning_rate': learning_rate
}

# Set N1 and N2 for 1D case
params['N1'] = n
params['N2'] = 1

# Compute the analytic solution
y_exact = pc_s_function(x0, 1, 1, alpha1, alpha2, value=value)

# Loop for different max_value
for max_value in [0, 0.1, 0.5, 1, 10, 100]: 

    # Model configurations
    configurations = [{'hidden_layers': [64,128], 'activations': ['relu','relu'], 
                       'l2_reg': 0, 'max_value': max_value, 
                       'use_kernel_constraint': True, 'use_bias_constraint': True}
    ]
    # Define the path for saving results
    path = f'{d}d/test_boundedweights/{problem}'

    # Loop through each configuration
    for config in configurations:
        hidden_layers = config['hidden_layers']
        activations = config['activations']
        l2_reg = config['l2_reg']
        max_value = config['max_value']
        use_kernel_constraint = config['use_kernel_constraint']
        use_bias_constraint = config['use_bias_constraint']

        # Create a description for the configuration
        config_desc = f'layers_{"-".join(map(str, hidden_layers))}_acts_{"-".join([act if act else "None" for act in activations])}_l2_{l2_reg}'
        log_dir = f'{path}/{config_desc}/rule{rule}/learning_rate{learning_rate}/max_value{max_value}'
        
        # Remove the directory if it exists
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        
        # Set the log directory in params
        params['log_dir'] = log_dir

        # Use a custom logger to capture output
        with imgNN.DualLogger(log_dir):
            # Plot and save the observation
            plt.figure(dpi=600)
            plt.plot(x0, pc_function(x0, value=value), label='True Function')
            plt.plot(x0, g, label='Noisy signal', linestyle='--')
            if problem == 'inpainting':
                plt.plot(D, mask[int(n / 2) - 100:int(n / 2) + 100], linewidth=5)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
            plt.savefig(f'{log_dir}/observation.png', bbox_inches='tight')
            plt.close()

            
            # Instantiate the model and optimizer
            model = imgNN.SimpleNN(hidden_layers, activations, l2_reg, max_value, use_kernel_constraint, use_bias_constraint)
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            try:
                # Train the model using the provided training function
                best_model, best_image, error_data = imgNN.train(model, optimizer, x0, g, params, T, solution=y_exact)
             
                # Generate test data for plotting
                x_test = np.linspace(0, 2, n).reshape(-1, 1).astype(np.float32)
                y_test = pc_function(x0, value=value)
                
                # Get the model's predictions on the test data
                y_pred = best_model(tf.convert_to_tensor(x0), training=False).numpy()
                
                # Plot the original, predicted, and analytic solution signals
                plt.figure(dpi=600)
                plt.plot(x0, y_test, label='Original signal')
                plt.plot(x0, y_pred, label='NN approximation', linestyle='--')
                plt.plot(x0, y_exact, label='Analytic solution', linestyle=':')
                if problem == 'inpainting':
                    plt.plot(D, mask[int(n / 2) - 100:int(n / 2) + 100], linewidth=5)
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
                plt.savefig(f'{log_dir}/reconstruction.png', bbox_inches='tight')
                plt.show()
                plt.close()
                
                # Compute custom loss components
                total_loss, l2_term, l1_term, TV_term = convert_tensors_to_numpy(
                    imgNN.custom_loss(g, y_pred, d, T, parameter, alpha2=alpha2, 
                                      alpha1=alpha1, alphaTV=alphaTV, 
                                      h=h, rule=rule)
                )
                
                # Print final loss and individual loss components
                print('')
                print(f" Final, Total Loss: {total_loss}")
                print(f"  L^2-data Loss (weighted): {alpha2 * l2_term}")
                print(f"  L^1-data Loss (weighted): {alpha1 * l1_term}")
                print(f"  TV-Loss (weighted): {alphaTV * TV_term}")
                print(f"  L^2 Loss: {l2_term}")
                print(f"  L^1 Loss: {l1_term}")
                print(f"  lossTV: {TV_term}")
                
                # Add additional parameters for saving
                params.update({
                    'total_loss': total_loss,
                    'l2_term': l2_term,
                    'l1_term': l1_term,
                    'TV_term': TV_term,
                    'hidden_layers': hidden_layers,
                    'activations': activations,
                    'l2_reg': l2_reg,
                    'max_value': max_value,
                    'use_kernel_constraint': use_kernel_constraint,
                    'use_bias_constraint': use_bias_constraint
                })
                
                # Save the updated parameters to files
                imgNN.save_parameters(log_dir, params)
                save_essential_data(log_dir=log_dir,params=params,best_image=best_image,error_data=error_data,g=g)
            except Exception as e:
                # Log any exceptions that occur during training
                error_log_file = os.path.join(log_dir, 'error.log')
                with open(error_log_file, 'w') as f:
                    f.write(str(e))
                    f.write("\n")
                    f.write(traceback.format_exc())  # Add traceback information
                print(f"Error occurred for configuration {config_desc}. Logged error and continuing.")
                print(str(e))

# Plot the original and analytic solution signals
plt.plot(x0, pc_function(x0, value=value), label='Original signal')
plt.plot(x0, y_exact, label='Analytic solution', linestyle='--')
plt.legend()
plt.show()


print('')
print(f" Analytic, Total Loss: {E(1, 1, alpha1, alpha2)}")
print(f"  L^2-data Loss (weighted): {E2(1, 1, alpha1, alpha2)}")
print(f"  L^1-data Loss (weighted): {E1(1, 1, alpha1, alpha2)}")
print(f"  L1 Grad Loss (weighted): {ETV(1, 1, alpha1, alpha2)}")
