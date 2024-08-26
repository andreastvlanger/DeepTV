#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 12:36:50 2024

MIT License

@authors: Andreas Langer, Sara Behnamian
"""
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import sys
import pickle
import shutil
import NN_image_processing as imgNN
import image_functions
import traceback  

# Define a params dictionary with default values
params = {
    #problem parameters
    'd': 2,  # Dimension of the problem (2D in this case)
    'image_nr': 0,  # Image number for selecting specific image input
    # 'height': 256,  # Default image height
    # 'width': 256,   # Default image width
    'r': 1/4,  # Radius parameter for the problem
    'problem': 'denoising',  # Problem type: can be 'inpainting', 'deblurring', or 'denoising'
    'blur_para': [10, 20.0],  # Parameters for blurring
    'sm_p': 'g',  # Smoothing parameter
    'seed_nr': 0,  # Seed number for noise generation
    'noise': 'Gauss_sp',  # Noise type: can be 'Gauss', 'Gauss_sp', 'Gauss_rv', 'Gauss_sp_rv', 'sp', 'sp_rv', 'rv'
    'gnoise': (0.0, 0.0),  # Gaussian noise parameters (mean, std)
    'spnoise': 0.0,  # Salt and pepper noise parameter
    'sp_ratio': 0.5,  # Salt to pepper ratio
    'rvnoise': 0,  # Random impulse noise parameter
    #model parameters
    'alpha1': 1.0,  # Regularization parameter for L1 norm
    'alpha2': 7.0,  # Regularization parameter for L2 norm
    'alpha_TV': 1.0,  # Regularization parameter for Total Variation norm
    'tv_type': 'TV2',  # Type of Total Variation: 'TV1' or 'TV2', 'TV21' (only for d=2),
    'tv2_smoothing': 'Huber', # Smoothing method for TV2: 'Huber', 'SmoothingOption1', 'SmoothingOption2'
    'gamma': 1e-10,  # Gamma value for TV2 smoothing
    'boundary_condition': 'Dirichlet',  # Boundary condition: 'Dirichlet' or 'Neumann'
    'rule': 'AR', # quadrature rule. choose from 'AR' (default) and 'TR' (composite trapezoid rule)
    'h': None,  # This will be calculated based on image size 
    #'weight_l2': 0.0,  # L2 weight regularization parameter !not needed!
    #optimizer parameter
    'learning_rate': 1e-3, # Learning rate for the optimizer
    #'batch_size': 100, #not needed for our work!!!!
    'epochs': 300001, # Number of training epochs 300001
    #saving
    'log_dir': 'GammaConvergence',  # Directory to save plots and logs
    'save_plots': True,  # Whether to save plots or not
    'plot_interval': 100,  # Interval to save plots
    #NN parameters
    'use_kernel_constraint': True,  # Whether to use kernel constraints
    'use_bias_constraint': True,  # Whether to use bias constraints
    'l2_reg': 0,  # L2 regularization strength of the weights
    'max_value': 100,  # Max value for kernel and bias constraints
    'configurations': [
        # Each configuration consists of: (hidden_layers, activations, l2_reg, max_value, use_kernel_constraint, use_bias_constraint)
        ([128, 128, 128], ['relu', 'relu', 'relu'], 'l2_reg', 'max_value', 'use_kernel_constraint', 'use_bias_constraint')
    ]
}

def save_essential_data(log_dir, **kwargs):
    file_path = os.path.join(log_dir, 'essential_data.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(kwargs, f)
    print(f"Saved essential data to {file_path}")
    print(f"Saved variables: {', '.join(kwargs.keys())}")      

# Define c1 and c2 functions
c1 = lambda alpha1, alpha2, alphaTV, r: 0  # Dirichlet boundary conditions
c2 = lambda alpha1, alpha2, alphaTV, r, value=1: (value - 
                (np.pi * r * alphaTV / (alpha2 * np.pi * r**2) - alpha1 / (2 * alpha2)))

# Define true_solution_circle function
def true_solution_circle(c1, c2, height, width, r):
    """
    Create a circular region in the image with different intensity values.

    Args:
        c1 (float): Intensity value outside the circle.
        c2 (float): Intensity value inside the circle.
        height (int): Height of the image.
        width (int): Width of the image.
        r (float): Radius of the circle relative to the image size.

    Returns:
        np.ndarray: Image with the circular region.
    """
    image = c1 * np.ones((height, width))
    center = (height // 2, width // 2)
    radius = height * r
    yc, xc = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((xc - center[1])**2 + (yc - center[0])**2)
    circle = dist_from_center <= radius
    image[circle] = c2
    return image

# Define energy functions
E = lambda c1, c2, alpha1, alpha2, alphaTV, r, value=1: (
    alpha1 * np.pi * r**2 * np.abs(value - c2) +
    alpha1 * (1 - np.pi * r**2) * np.abs(c1) +
    alpha2 * np.pi * r**2 * np.square(value - c2) +
    alpha2 * (1 - np.pi * r**2) * np.square(c1) +
    2 * np.pi * r * alphaTV * np.abs(c2 - c1)
)

E1 = lambda c1, c2, alpha1, alpha2, alphaTV, r, value=1: (
    alpha1 * np.pi * r**2 * np.abs(value - c2) +
    alpha1 * (1 - np.pi * r**2) * np.abs(c1)
)

E2 = lambda c1, c2, alpha1, alpha2, alphaTV, r, value=1: (
    alpha2 * np.pi * r**2 * np.square(value - c2) +
    alpha2 * (1 - np.pi * r**2) * np.square(c1)
)

ETV = lambda c1, c2, alpha1, alpha2, alphaTV, r, value=1: (
    2 * np.pi * r * alphaTV * np.abs(c2 - c1)
)

# For Gamma-convergence experiments we need Dirichlet boundary conditions
d = params['d']

image_nr = params['image_nr']

# Start of the nested loops
for tv_type in ['TV2', 'TV21']:
    params['tv_type'] = tv_type
    
    tv2_smoothing_options = ['Huber', 'SmoothingOption1', 'SmoothingOption2']
    
    for tv2_smoothing in tv2_smoothing_options:
        params["tv2_smoothing"] = tv2_smoothing
        
        for n in [2**i for i in range(5, 11)]:  # 2^5 to 2^10
            if image_nr == 0:
                if n % 2 == 0:
                    n += 1  # Use odd numbers so that the middle exists
                height, width = n, n
                image = 0. * np.ones((height, width))
                center = (height // 2, width // 2)
                radius = height * params['r']  # radius r = 1/4
                yc, xc = np.ogrid[:height, :width]
                dist_from_center = np.sqrt((xc - center[1])**2 + (yc - center[0])**2)
                circle = dist_from_center <= radius
                image[circle] = 1
            else:
                image = image_functions.input_image(image_nr)  # If input_image() is used, it should be added to the folder-path

            y = image.astype(np.float32)
            N1, N2 = y.shape
            params['N1'] = N1
            params['N2'] = N2
            # Scale to [0,1] x [0,1]

            # Calculate h and update params
            hx = 1 / N1 
            hy = 1 / N2
            params['h'] = hx  # Assuming square images
            params['hx'] = hx
            params['hy'] = hy

            print(f'h = {hx}, {hy};\n N1 = {N1}, N2 = {N2}')
            # Problem selection
            problem = params['problem']
            if problem == 'deblurring':
                blur_para = params['blur_para']
                sm_p = params['sm_p']
                parameter = {'blur_para': blur_para, 'height': N1, 'width': N2, 'sm_p': sm_p}
                T = imgNN.T_blurring
            elif problem == 'inpainting':
                mask = imgNN.create_mask(N1, N2)
                parameter = mask
                T = imgNN.T_inpainting
            else:
                parameter = ()
                T = imgNN.T_denoising

            y_vec = np.reshape(y, (N1 * N2, 1))
            g_vec = T(y_vec, parameter)
            g = np.reshape(g_vec, (N1, N2))

            # Noise addition
            noise = params['noise']
            path = f'{d}d/{problem}/size{N1}x{N2}/image{image_nr}/{noise}'
            if noise in ['Gauss', 'Gauss_sp', 'Gauss_rv', 'Gauss_sp_rv']:
                gnoise = params['gnoise']
                seed_nr = params['seed_nr']
                g = image_functions.add_gauss_noise(g, gnoise, seed_nr, c='real').astype(np.float32)
                path = f'{path}/gnoise{gnoise}'

            if noise in ['sp', 'Gauss_sp', 'sp_rv', 'Gauss_sp_rv']:
                spnoise = params['spnoise']
                sp_ratio = params['sp_ratio']
                g = image_functions.add_salt_pepper_noise(g, spnoise, sp_ratio, seed_nr=0)
                path = f'{path}/spnoise{spnoise}'

            if noise in ['rv', 'sp_rv', 'Gauss_rv', 'Gauss_sp_rv']:
                rvnoise = params['rvnoise']
                g = image_functions.add_random_impulse_noise(g, rvnoise, seed_nr=0)
                path = f'{path}/rvnoise{rvnoise}'


            tensor = tf.linspace(0.0, float(g.shape[0] - 1), g.shape[0]) / N1
            x_tf = tf.reshape(tensor, (-1, 1))
            y_tf = tf.reshape(tensor, (-1, 1))
            X, Y = tf.meshgrid(x_tf[:, 0], y_tf[:, 0])
            xy = tf.stack([tf.reshape(X, [-1]), tf.reshape(Y, [-1])], axis=1)
            g_vec = tf.reshape(g, [N1 * N2, 1])


            model = None
            optimizer = None

            # Calculate the true solution
            u = true_solution_circle(c1(params['alpha1'], params['alpha2'], params['alpha_TV'], params['r']), 
                                    c2(params['alpha1'], params['alpha2'], params['alpha_TV'], params['r']), 
                                    height, width, params['r'])
            u_vec = np.reshape(u, (N1 * N2, 1)).astype(np.float32)

            # Save the original stdout
            original_stdout = sys.stdout
            #for max_value in [0, 1, 10, 100, 1000, 10000]:
            for max_value in [10000]:
                params["max_value"] = max_value
                for (hidden_layers, activations, l2_reg_key, max_value_key, use_kernel_constraint_key, use_bias_constraint_key) in params['configurations']:
                    l2_reg = params[l2_reg_key]
                    max_value = params[max_value_key]
                    use_kernel_constraint = params[use_kernel_constraint_key]
                    use_bias_constraint = params[use_bias_constraint_key]

                    # Create a descriptive folder name
                    config_desc = f'layers_{"-".join(map(str, hidden_layers))}_acts_{"-".join([act if act else "None" for act in activations])}'
                    log_dir = f'GammaConvergence/{config_desc}/{params["tv_type"]}/{params["tv2_smoothing"]}/size{N1}x{N2}_maxvalue{max_value}'
                    params['log_dir'] = log_dir
                    if os.path.exists(log_dir):
                        shutil.rmtree(log_dir)
                    os.makedirs(log_dir, exist_ok=True)
                    
                    with imgNN.DualLogger(log_dir):
                        
                        # Save the original image (ground truth)
                        plt.figure() 
                        plt.axis('off')
                        plt.imshow(y, cmap='gray', vmin=0, vmax=1)
                        plt.title('Original Image')
                        plt.savefig(f"{log_dir}/original_image.png")
                        plt.close()

                        # Save the observation image
                        plt.axis('off')
                        plt.imshow(g, cmap='gray')
                        plt.title('Observation')
                        plt.savefig(f'{log_dir}/observation.png')
                        plt.show()
                        plt.close()

                        # Define the model and optimizer
                        model = imgNN.SimpleNN(hidden_layers, activations, l2_reg, max_value=max_value, use_kernel_constraint=use_kernel_constraint, use_bias_constraint=use_bias_constraint)
                        optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
                        test = model(xy)
                        
                        try:
                            # Train the model
                            best_model, best_image, error_data = imgNN.train(model, optimizer, xy, g_vec, params, T, solution=u_vec)
                           
                            ############### Extract the best loss and corresponding epoch
                            best_loss_list = error_data['best_loss_list']
                            best_loss_epoch_list = error_data['best_loss_epoch_list']
                            params['best_loss'] = best_loss_list[-1]
                            params['best_loss_epoch'] = best_loss_epoch_list[-1]
                            #####################

                            # Save the parameters used for this configuration
                            imgNN.save_parameters(log_dir, params)  # Save both .tex and .pkl formats
                            save_essential_data(log_dir=log_dir,params=params,best_image=best_image,error_data=error_data,g=g)
                            # Ensure the best weights are used for the final reconstruction
                            best_model.set_weights(best_model.get_weights())
                            y_pred = best_model(xy, training=False).numpy()

                            # Save the final reconstruction
                            plt.axis('off')
                            plt.imshow(np.reshape(y_pred, (N1, N2)), cmap='gray', vmin=0, vmax=1)
                            plt.title('Final Reconstruction')
                            plt.savefig(f'{log_dir}/reconstruction.png')
                            plt.close()

                            # Save the best image from training for comparison
                            plt.axis('off')
                            plt.imshow(np.reshape(best_image, (N1, N2)), cmap='gray', vmin=0, vmax=1)
                            plt.title('Best Training Image')
                            plt.savefig(f'{log_dir}/best_training_image.png')
                            plt.show()
                            plt.close()

                            # Calculate and print quality measures
                            print('PSNR, SSIM, MAE: ', image_functions.calculate_quality_measures(y, np.reshape(best_image, (N1, N2))))

                            # Calculate the true solution
                            u = true_solution_circle(c1(params['alpha1'], params['alpha2'], params['alpha_TV'], params['r']), 
                                                    c2(params['alpha1'], params['alpha2'], params['alpha_TV'], params['r']), 
                                                    height, width, params['r'])
                            u_vec = np.reshape(u, (N1 * N2, 1)).astype(np.float32)
                            
                            diff1 = hx * hy * params['alpha1'] * np.sum(np.abs(u_vec - y_pred))
                            diff2 = hx * hy * params['alpha2'] * np.sum(np.square(u_vec - y_pred))
                            
                            print(f'Diff1: {diff1}')
                            print(f'Diff2: {diff2}')
                            
                            # Save the analytic solution plot
                            plt.figure()
                            plt.axis('off')
                            plt.imshow(u, cmap='gray', vmin=0, vmax=1)
                            plt.title('Analytic solution')
                            plt.savefig(f'{log_dir}/analytic_solution.png')
                            plt.show()
                            plt.close()

                            # Print analytic solution values
                            print(c1(params['alpha1'], params['alpha2'], params['alpha_TV'], params['r']),
                                c2(params['alpha1'], params['alpha2'], params['alpha_TV'], params['r']))

                            print(f' Analytic energy: {E(c1(params["alpha1"], params["alpha2"], params["alpha_TV"], params["r"]), c2(params["alpha1"], params["alpha2"], params["alpha_TV"], params["r"]), params["alpha1"], params["alpha2"], params["alpha_TV"], params["r"])}')

                            print(f'Analytic l2: {E2(c1(params["alpha1"], params["alpha2"], params["alpha_TV"], params["r"]), c2(params["alpha1"], params["alpha2"], params["alpha_TV"], params["r"]), params["alpha1"], params["alpha2"], params["alpha_TV"], params["r"])}')
                            print(f'Analytic l1: {E1(c1(params["alpha1"], params["alpha2"], params["alpha_TV"], params["r"]), c2(params["alpha1"], params["alpha2"], params["alpha_TV"], params["r"]), params["alpha1"], params["alpha2"], params["alpha_TV"], params["r"])}')
                            print(f'Analytic tv: {2 * np.pi * params["r"] * np.abs(c1(params["alpha1"], params["alpha2"], params["alpha_TV"], params["r"]) - c2(params["alpha1"], params["alpha2"], params["alpha_TV"], params["r"]))}')

                        except Exception as e:
                            # Save error information to log
                            error_log_file = os.path.join(log_dir, 'error.log')
                            with open(error_log_file, 'w') as f:
                                f.write(str(e))
                                f.write("\n")
                                f.write(traceback.format_exc())  # Add traceback information
                            print(f"Error occurred for configuration {config_desc}. Logged error and continuing.")
                            print(str(e))  # Print the error to the terminal as well
