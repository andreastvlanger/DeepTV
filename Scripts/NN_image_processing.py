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

Created on Sat Jun  8 13:11:00 2024

GNU GENERAL PUBLIC LICENSE Version 3

@authors: Andreas Langer, Sara Behnamian
"""
import os
import sys
import pickle
import numpy as np
import tensorflow as tf
#import matplotlib  # if you dont want to see the plots 
#matplotlib.use('Agg')  # Use a non-GUI backend for generating plots
import matplotlib.pyplot as plt
######################
# own functions
import smooth_filter as sf
import image_functions
####################

        
params = {
    #problem parameters
    'd': 2,  # Choose dimension: 1 or 2
    'image_nr': 0,
    'height': 96, #only for 'image_nr'=0, otherwise specified by the image!!!!
    'width': 96, #only for 'image_nr'=0, otherwise specified by the image!!!!
    'r': 1/4,  # Radius parameter for the problem 'image_nr'=0
    'N1': None,  # N1 dimension
    'N2': None,  # N2 dimension
    'problem': 'denoising',  # Choose from 'inpainting', 'deblurring', 'denoising'
    'blur_para': [10, 20.0], # [kernel-size, standard deviation]: kernel-size creates a kernel of size kernel-size x kernel-size if kernel-size is odd, otherwise kernel-size+1 x kernel-size+1 (this ensures odd kernel-size)
    'sm_p': 'g', # type of filter: Choose from 'g' (Gaussian), 'm' (mean)
    'seed_nr': 0,  # Seed number for noise generation
    'noise': 'Gauss_sp',  # Choose from 'Gauss', 'Gauss_sp', 'Gauss_rv', 'Gauss_sp_rv', 'sp', 'sp_rv', 'rv'
    'gnoise': (0.0, 0.0), #(mean, standard variation)
    'spnoise': 0.0, #level of salt-and-pepper noise
    'sp_ratio': 0.5,# ratio between salt-and-pepper noise
    'rvnoise': 0, #level of random valued impulse noise
    #model parameters
    'alpha1': 1, #100.0,
    'alpha2': 7, #100.0,
    'alpha_TV': 1.0,
    'tv_type': 'TV21',  # Choose from 'TV2', 'TV21' (only for d=2),
    'tv2_smoothing': 'Huber',  # Choose from 'Huber', 'SmoothingOption1', 'SmoothingOption2'
    'gamma': 1e-10,  # Gamma value for TV2 smoothing
    'boundary_condition': 'Dirichlet',  # Choose from 'Dirichlet', 'Neumann'
    'rule': 'AR', # quadrature rule. choose from 'AR' (default) and 'TR' (composite trapezoid rule)
    'h': None,  # This will be calculated based on image size 
    #optimizer parameter
    'learning_rate': 1e-3,
    'epochs': 10001,
    #saving
    'log_dir': 'logs',
    'save_plots': True,  # Whether to save plots or not
    'plot_interval': 100,  # Interval to save plots
    #NN parameters
    'use_kernel_constraint': True,  # Whether to use kernel constraints
    'use_bias_constraint': True,  # Whether to use bias constraints
    'l2_reg': 0,  # L2 regularization strength of the weights
    'max_value': 1,  # Max value for kernel and bias constraints
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

# Save the original stdout so we can restore it later
original_stdout = sys.stdout

class DualLogger:
    def __init__(self, log_dir, log_file='output.log'):
        self.log_dir = log_dir
        self.log_file = log_file
        self.original_stdout = sys.stdout
        self.log_file_path = os.path.join(self.log_dir, self.log_file)

    def __enter__(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.log_file = open(self.log_file_path, 'w')
        sys.stdout = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        self.log_file.close()

    def write(self, message):
        self.original_stdout.write(message)
        self.log_file.write(message)
        self.flush()  # Ensure immediate write to the file

    def flush(self):
        self.original_stdout.flush()
        self.log_file.flush()
    
    
def save_parameters(log_dir, params):
    # Save parameters to .txt file
    param_file_txt = os.path.join(log_dir, 'parameters.txt')
    with open(param_file_txt, 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    print(f"Saved parameters to {param_file_txt}")

    # Save parameters to .tex file
    param_file_tex = os.path.join(log_dir, 'parameters.tex')
    with open(param_file_tex, 'w') as f:
        f.write("% Auto-generated parameters file\n")
        f.write("\\providecommand{\\Data}[1]{\n")
        f.write("    \\csname Data/#1\\endcsname\n")
        f.write("}\n\n")
        for key, value in params.items():
            if isinstance(value, (list, tuple)):
                value_str = ', '.join(map(str, value))
                #f.write(f"\\newcommand{{\\{key}}}{{{value_str}}}\n")
                f.write(f"\\expandafter\\def\\csname Data/\\DataPrefix/{key}\\endcsname{{\\pgfmathprintnumber{{{value_str}}}}}\n")
            else:
                #f.write(f"\\newcommand{{\\{key}}}{{{value}}}\n")
                f.write(f"\\expandafter\\def\\csname Data/\\DataPrefix/{key}\\endcsname{{\\pgfmathprintnumber{{{value}}}}}\n")
    print(f"Saved parameters to {param_file_tex}")

    # Save parameters to .pkl file
    param_file_pkl = os.path.join(log_dir, 'parameters.pkl')
    with open(param_file_pkl, 'wb') as f:
        pickle.dump(params, f)
    print(f"Saved parameters to {param_file_pkl}")


class SimpleNN(tf.keras.Model):
    def __init__(self, hidden_layers, activations, l2_reg=1e-4, max_value=100, use_kernel_constraint=True, use_bias_constraint=True):
        super(SimpleNN, self).__init__()
        self.hidden_layers = []
        for neurons, activation in zip(hidden_layers, activations):
            kernel_constraint = tf.keras.constraints.MaxNorm(max_value=max_value) if use_kernel_constraint else None
            bias_constraint = tf.keras.constraints.MaxNorm(max_value=max_value) if use_bias_constraint else None
            
            self.hidden_layers.append(tf.keras.layers.Dense(
                neurons, 
                activation=activation, 
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                kernel_constraint=kernel_constraint, 
                bias_constraint=bias_constraint
            ))
        
        kernel_constraint = tf.keras.constraints.MaxNorm(max_value=max_value) if use_kernel_constraint else None
        bias_constraint = tf.keras.constraints.MaxNorm(max_value=max_value) if use_bias_constraint else None
        
        self.output_layer = tf.keras.layers.Dense(
            1,
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint
        )

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

def numerical_integration_1D(u, h, rule='TR'):
    u_reshaped = tf.reshape(u, [-1])
    if rule == 'TR':  # Composite Trapezoid Rule
        return h / 2 * (u_reshaped[0] + u_reshaped[-1]) + h * tf.reduce_sum(u_reshaped[1:-1])
    elif rule == 'AR':  # Composite Midpoint Rule
        return h * tf.reduce_sum(u_reshaped)
    else:
        return None

def numerical_integration_2D(u, hx, hy, m1, m2, rule='AR'):
    if rule == 'TR':  # Composite Trapezoid Rule
        u_reshaped = tf.reshape(u, [m1, m2])
        return hx * hy / 4 * (
            u_reshaped[0, 0] + u_reshaped[0, -1] + u_reshaped[-1, 0] + u_reshaped[-1, -1] +
            2 * tf.reduce_sum(u_reshaped[1:-1, 0]) + 2 * tf.reduce_sum(u_reshaped[1:-1, -1]) +
            2 * tf.reduce_sum(u_reshaped[0, 1:-1]) + 2 * tf.reduce_sum(u_reshaped[-1, 1:-1]) +
            4 * tf.reduce_sum(u_reshaped[1:-1, 1:-1]))
    elif rule == 'AR':
        return hx * hy * tf.reduce_sum(u)
    else:
        return None

def get_dimensions(y_pred, d):
    if d == 1:
        m1 = y_pred.shape[0]
        m2 = 1
    elif d == 2:
        m1 = int(np.sqrt(y_pred.shape[0]))
        m2 = m1
    return m1, m2

def compute_gradient(p, x, h, method='FD'):
    if method == 'Cont':
        x = tf.convert_to_tensor(x)
        with tf.GradientTape() as tape:
            tape.watch(x)
            p = model(x, training=True)
        grad_p = tape.gradient(p, x)
    elif method == 'FD':
        grad_p = tf.subtract(p[1:], p[:-1]) / h
        zero_tensor = tf.constant([[0]], dtype=grad_p.dtype)
        grad_p = tf.concat([grad_p, zero_tensor], axis=0)
    else:
        print('Method not implemented!')
        return None
    return grad_p

def HuberRegularization(dydx, dydy, gamma):  # only needed for 2d
    norm = tf.sqrt(tf.square(dydx) + tf.square(dydy))
    condition = norm > gamma
    
    result = tf.where(condition, norm - 1/2*gamma, 
                      1/(2*gamma) * (tf.square(dydx) + tf.square(dydy)))
    return result
        
def HuberRegularizationTV21(dydx_fd, dydy_fd, dydx_bd, dydy_bd, gamma):  # only needed for 2d
    norm12 = tf.square(dydx_fd) + tf.square(dydy_fd)
    norm1 = tf.sqrt(norm12) 
    norm22=tf.square(dydx_bd) + tf.square(dydy_bd)
    norm2 = tf.sqrt(norm22)
    norm = 0.5 * (norm1 + norm2)
    condition1 = norm1 > gamma
    condition2 = norm2 > gamma
    
    return tf.where(condition1,
             tf.where(condition2,norm - 0.5*gamma, 0.5*norm1 - gamma/4 + 1/(4*gamma)*norm22),
             tf.where(condition2,0.5*norm2 - gamma/4 + 1/(4*gamma)*norm12,1/(4*gamma)*(norm12+norm22)))
    

def custom_loss(y_true, y_pred, d, T, parameter, alpha2=1.0, alpha1=0.1, 
                alphaTV=1.0, h=1, rule='AR', tv_type='TV2', 
                boundary_condition='Neumann', tv2_smoothing='Huber', gamma=1e-10):
    if d == 1:
        m1 = y_pred.shape[0]
        m2 = 1
        hx = h
        hy = h
        # L2-data term
        l2_term = numerical_integration_1D(tf.square(y_true - T(y_pred, parameter)), h, rule)
        # L1-data term
        l1_term = numerical_integration_1D(tf.abs(y_true - T(y_pred, parameter)), h, rule)
        
        # FD gradient
        dy = tf.subtract(y_pred[1:], y_pred[:-1]) / h
        if boundary_condition == 'Neumann':
            zero_tensor = tf.constant([[0]], dtype=dy.dtype)
        elif boundary_condition == 'Dirichlet':
            zero_tensor = tf.constant([[y_pred[-1]]], dtype=dy.dtype) / h
        dy = tf.concat([dy, zero_tensor], axis=0)
        
        # TV-term
        TV_term = numerical_integration_1D(tf.abs(dy), h, rule)
        
    elif d == 2:
        m1 = int(np.sqrt(y_pred.shape[0]))
        m2 = m1
        hx = h
        hy = h
        l2_term = numerical_integration_2D(tf.square(y_true - T(y_pred, parameter)), hx, hy, m1, m2, rule)
        l1_term = numerical_integration_2D(tf.abs(y_true - T(y_pred, parameter)), hx, hy, m1, m2, rule)
        
        # FD gradient
        ys = tf.reshape(y_pred, [m1, m2])
        dydx = tf.subtract(ys[1:, :], ys[:-1, :]) / hx
        dydy = tf.subtract(ys[:, 1:], ys[:, :-1]) / hy
                
        if boundary_condition == 'Neumann':  # homogeneous Neumann
            zero_tensor_x = tf.zeros((1, m2), dtype=ys.dtype)
            zero_tensor_y = tf.zeros((m1, 1), dtype=ys.dtype)
        elif boundary_condition == 'Dirichlet':  # homogeneous Dirichlet
            zero_tensor_x = tf.reshape(ys[-1, :] / hx, (1,-1))
            zero_tensor_y = tf.reshape(ys[:, -1] / hy, (-1,1))
        
        dydx1 = tf.concat([dydx, zero_tensor_x], axis=0)
        dydy1 = tf.concat([dydy, zero_tensor_y], axis=1)
        
        # Backward Differences: gradient
        dydx_bd = tf.subtract(ys[:-1, :], ys[1:, :]) / hx
        dydy_bd = tf.subtract(ys[:, :-1], ys[:, 1:]) / hy
        
        if boundary_condition == 'Neumann':  # homogeneous Neumann
            zero_tensor_x_bd = tf.zeros((1, m2), dtype=ys.dtype)
            zero_tensor_y_bd = tf.zeros((m1, 1), dtype=ys.dtype)
        elif boundary_condition == 'Dirichlet':  # homogeneous Dirichlet
            zero_tensor_x_bd = tf.reshape(ys[1, :] / hx, (1,-1))
            zero_tensor_y_bd = tf.reshape(ys[:, 1] / hy, (-1,1))
        
        dydx_bd1 = tf.concat([zero_tensor_x_bd, dydx_bd], axis=0)
        dydy_bd1 = tf.concat([zero_tensor_y_bd, dydy_bd], axis=1)
        
        if tv_type == 'TV2':  # TV_2 isotropic
            if tv2_smoothing == 'Huber':
                dy_abs = HuberRegularization(dydx1, dydy1, gamma)
            elif tv2_smoothing == 'SmoothingOption1':
                dy_abs = tf.sqrt(tf.square(dydx1) + tf.square(dydy1) + gamma)
            elif tv2_smoothing == 'SmoothingOption2':
                dy_abs = tf.maximum(tf.sqrt(tf.square(dydx1) + tf.square(dydy1)), gamma)
            else:
                dy_abs = tf.sqrt(tf.square(dydx1) + tf.square(dydy1))
        elif tv_type == 'TV21':  # TV_21 -> forward backward gradient
            if tv2_smoothing == 'Huber':  
                dy_abs = HuberRegularizationTV21(dydx1, dydy1, dydx_bd1, dydy_bd1, gamma)
            elif tv2_smoothing == 'SmoothingOption1':
                dy_abs = 0.5 * (tf.sqrt(tf.square(dydx1) + tf.square(dydy1) + gamma) +
                                tf.sqrt(tf.square(dydx_bd1) + tf.square(dydy_bd1) + gamma))
            elif tv2_smoothing == 'SmoothingOption2':
                dy_abs =  0.5 *(tf.maximum(tf.sqrt(tf.square(dydx1) + tf.square(dydy1)),gamma)
                                           + tf.maximum(tf.sqrt(tf.square(dydx_bd1) + tf.square(dydy_bd1)), gamma))
            else:
                dy_abs = 0.5 * (tf.sqrt(tf.square(dydx1) + tf.square(dydy1)) 
                                + tf.sqrt(tf.square(dydx_bd1) + tf.square(dydy_bd1)))
        
        TV_term = numerical_integration_2D(dy_abs, hx, hy, m1, m2, rule)
    
    total_loss = (alpha2 * l2_term +
                  alpha1 * l1_term + 
                  alphaTV * TV_term)
    
    return total_loss, l2_term, l1_term, TV_term

class Trainer:
    def __init__(self, model, optimizer, params, T, parameter):
        self.model = model
        self.optimizer = optimizer
        self.d = params['d']
        self.parameter = parameter
        self.T = T
        self.weight_mse = params['alpha2']
        self.alpha1 = params['alpha1']
        self.alphaTV = params['alpha_TV']
        self.h = params['h']
        self.tv_type = params['tv_type']
        self.rule = params['rule']
        self.boundary_condition = params['boundary_condition']
        self.tv2_smoothing = params['tv2_smoothing']
        self.gamma = params['gamma']

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            tape.watch(x)
            predictions = self.model(x, training=True)
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                predictions = self.model(x, training=True)
            total_loss, l2_term, l1_term, TV_term = custom_loss(
                y, predictions, self.d, self.T, self.parameter, self.weight_mse, 
                self.alpha1, self.alphaTV, self.h, rule=self.rule, tv_type=self.tv_type,
                boundary_condition=self.boundary_condition, tv2_smoothing=self.tv2_smoothing, gamma=self.gamma)
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        if self.d == 2:
            clipped_gradients, global_norm = tf.clip_by_global_norm(gradients, 1.0)
            gradients = clipped_gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return total_loss, l2_term, l1_term, TV_term

def save_combined_error_plots1d(plots1, plots2, label1, label2, idx, log_dir, filename='combined_error_plot'):
    fig, axs = plt.subplots(10, 4, figsize=(20, 50))
    axs = axs.flatten()
    for i, (plot1, plot2) in enumerate(zip(plots1, plots2)):
        axs[i].plot(plot1, label=label1)
        axs[i].plot(plot2, label=label2)
        axs[i].axis('on')
        axs[i].legend()
    plt.tight_layout()
    plt.savefig(f'{log_dir}/{filename}_{idx}.png')
    plt.close()

def save_combined_plots1d(plots, x_train, idx, log_dir, filename='combined_plot'):
    fig, axs = plt.subplots(10, 4, figsize=(20, 50))
    axs = axs.flatten()
    for i, plot in enumerate(plots):
        axs[i].plot(x_train, plot)
        axs[i].axis('on')
    plt.tight_layout()
    plt.savefig(f'{log_dir}/{filename}_{idx}.png')
    plt.close()

def save_combined_plots(plots, idx, log_dir):
    fig, axs = plt.subplots(10, 4, figsize=(20, 50))
    axs = axs.flatten()
    for i, plot in enumerate(plots):
        axs[i].imshow(plot, cmap='gray', vmin=0, vmax=1)
        axs[i].axis('off')
    plt.tight_layout()
    plt.savefig(f'{log_dir}/combined_plot_{idx}.png')
    plt.close()

def train(model, optimizer, x_train, y_train, params, T, parameter=None, solution=None):
    trainer = Trainer(model, optimizer, params, T, parameter)  # Initialize Trainer with params
    best_loss = float('inf')
    best_weights = None
    best_image = None
    plots = []

    estimate_list = []
    estimate1_list = []
    estimate2_list = []
    real_error = []
    combined_plot_idx = 0
    best_loss_list=[] #List of best losses 
    best_loss_epoch_list=[] #List when the best loss is changed
    
    for epoch in range(params['epochs']):
        total_loss, l2_term, l1_term, TV_term = trainer.train_step(x_train, y_train)
        predictions = model(x_train, training=False)
        total_loss, l2_term, l1_term, TV_term = custom_loss(
            y_train, predictions, params['d'], T, parameter, 
            params['alpha2'], params['alpha1'], params['alpha_TV'], 
            params['h'], rule=params['rule'], tv_type=params['tv_type'], 
            boundary_condition=params['boundary_condition'], 
            tv2_smoothing=params['tv2_smoothing'], gamma=params['gamma']
        )

        current_total_loss = total_loss.numpy()
        if current_total_loss < best_loss:
            best_loss = current_total_loss
            best_loss_list.append(best_loss)
            best_loss_epoch_list.append(epoch)
            best_weights = model.get_weights()
            best_image = predictions.numpy()
            model.save_weights(f"{params['log_dir']}/best_weights.weights.h5")
            #model.save(f"{params['log_dir']}/best_model", save_format="tf")
            model.save(f"{params['log_dir']}/best_model.keras")
            N1, N2 = get_dimensions(predictions, params['d'])
            
            #if params['problem']=='denoising': #error estimates only if denoising
            estimate, estimate1, estimate2 = error_estimate(predictions, y_train, 
                                                T, parameter,params['alpha1'], 
                                                params['alpha2'], params['alpha_TV'], 
                                                params['h'], N1, N2, params['rule'], 
                                                params['d'], bc = params['boundary_condition'],
                                                tv_type=params['tv_type'])
            estimate_list.append(estimate)
            estimate1_list.append(estimate1)
            estimate2_list.append(estimate2)

            if params['d'] == 1:
                if solution is not None:
                    error = np.sqrt(numerical_integration_1D(np.square(solution - predictions), params['h']))
                    real_error.append(error)
                plots.append(predictions)
                if len(plots) == 40:
                    save_combined_plots1d(plots, x_train, combined_plot_idx, params['log_dir'])
                    combined_plot_idx += 1
                    plots = []
            elif params['d'] == 2:
                if solution is not None:
                    error = np.sqrt(numerical_integration_2D(np.square(solution - predictions), params['h'], params['h'], params['N1'], params['N2'], params['rule']))
                    real_error.append(error)
                u = np.reshape(predictions, (N1, N2))
                plots.append(u)
                if len(plots) == 40:
                    save_combined_plots(plots, combined_plot_idx, params['log_dir'])
                    combined_plot_idx += 1
                    plots = []

        if epoch % params['plot_interval'] == 0:
            print(f"Epoch {epoch}, Total Loss: {total_loss.numpy()}")
            print(f"  MSE Loss (weighted): {params['alpha2'] * l2_term.numpy()}")
            print(f"  L1 Loss (weighted): {params['alpha1'] * l1_term.numpy()}")
            print(f"  L1 Grad Loss (weighted): {params['alpha_TV'] * TV_term.numpy()}")
            print(f"  MSE Loss: {l2_term.numpy()}")
            print(f"  L1 Loss: {l1_term.numpy()}")
            print(f"  L1 Grad Loss: {TV_term.numpy()}")
            print(f"  Best Loss so far: {best_loss}")
            print(f"  Error estimate: {estimate_list[-1]}")
            if solution is not None:
                print(f"  Real error: {real_error[-1]}")
            print(".......................")
    if plots:
        if params['d'] == 1:
            save_combined_plots1d(plots, x_train, combined_plot_idx, params['log_dir'])
            if solution is not None:
                plt.figure(dpi=350)
                plt.plot(real_error, label='Real error', color='blue')
                plt.plot(estimate_list, label='Error estimate', linestyle='--', color='black')
                plt.plot(estimate1_list, label='Error estimate 1', linestyle=':', color='green')
                plt.plot(estimate2_list, label='Error estimate 2', linestyle=':', color='red')
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
                plt.savefig(f"{params['log_dir']}/error_plot.png", bbox_inches='tight')
                plt.show()
                plt.close()
        elif params['d'] == 2:
            save_combined_plots(plots, combined_plot_idx, params['log_dir'])
            if solution is not None:
                plt.figure(dpi=350)
                plt.plot(real_error, label='Real error', color='blue')
                plt.plot(estimate_list, label='Error estimate', linestyle='--', color='black')
                plt.plot(estimate1_list, label='Error estimate 1', linestyle=':', color='green')
                plt.plot(estimate2_list, label='Error estimate 2', linestyle=':', color='red')
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
                plt.savefig(f"{params['log_dir']}/error_plot.png", bbox_inches='tight')
                plt.show()
                plt.close()

    model.set_weights(best_weights)
    error_data = {
        'real_error': real_error,
        'estimate1_list': estimate1_list,
        'estimate2_list': estimate2_list,
        'estimate_list': estimate_list,
        'best_loss_list': best_loss_list,
        'best_loss_epoch_list': best_loss_epoch_list
    }
    return model, best_image, error_data

# def diff_operator(n, h):
#     a1 = [1/h] * (n - 1)
#     a0 = -1 * np.array(a1.copy(), dtype=np.float32)
#     a0 = np.append(a0, [0])
#     diags = [a0, a1]
#     offset = [0, 1]
#     GRAD = sp.diags(diags, offset)
#     DIV = -GRAD.T
#     return GRAD.astype(np.float32), DIV

def error_estimate(v, g, T, parameter, alpha1, alpha2, alphaTV, h, N1, N2, rule, 
                   d, bc='Neumann', tv_type='TV2'):
    #Only implemented for T=T_denoising (=I). 
    
    eps = 1e-10
    if d==2 and tv_type == 'TV21': #'TV21' is only allowed for d=2
        GRAD = image_functions.function_nabla_FB(N1, N2, h, h, bc=bc)
    else:
        GRAD = image_functions.function_nabla(N1, N2, h, h, bc=bc)
    DIV = (-GRAD.T).astype(np.float32)
    if d == 1:
        #estimate part 1
        Tv_g = T(v, parameter) - g
        estimate1 = 2 * alpha2 * tf.sqrt(numerical_integration_1D(tf.square(T(Tv_g, parameter)), h, rule)).numpy() #here would be adjoint
        estimate1 = 1 / alpha2 * estimate1.astype(np.float32) #1/alpha2 Only true for T=T_denoising!!!
        #estimate part 2
        nabla_v = GRAD @ v
        #tv_type not relevant in 1d as TV1 and TV2 are the same! 
        n_nabla_v = image_functions.function_eta(nabla_v, N1 * N2)  #why not np.abs()?
        p = (nabla_v / (n_nabla_v + eps)).astype(np.float32)
        div_p = -DIV @ p
        q = Tv_g / (np.abs(Tv_g) + eps)
        Tq = T(q, parameter)
        estimate2 = 1 / alpha2 * np.sqrt(numerical_integration_1D(np.square(alpha1 * Tq + alphaTV * div_p), h, rule))#1/alpha2 Only true for T=T_denoising!!!
    elif d == 2:
        #estimate part 1
        Tv_g = T(v, parameter) - g
        estimate1 = 1 / alpha2 * 2 * alpha2 * tf.sqrt(numerical_integration_2D(tf.square(T(Tv_g, parameter)), h, h, N1, N2, rule)).numpy()#here would be adjoint#1/alpha2 Only true for T=T_denoising!!!
        #estimate part 2
        nabla_v = GRAD @ v
        # Attention: function_eta depends on the choice of TV
        n_nabla_v = image_functions.function_eta(nabla_v, N1 * N2, tv_type=tv_type)
        if tv_type=='TV21':
            p = (nabla_v / (np.vstack([n_nabla_v, n_nabla_v, n_nabla_v, n_nabla_v]) + eps)).astype(np.float32)
        else:
            p = (nabla_v / (np.vstack([n_nabla_v, n_nabla_v]) + eps)).astype(np.float32)
        q = (Tv_g / (np.abs(Tv_g) + eps))
        Tq = T(q, parameter) #here would be adjoint
        div_p = -DIV @ p
        estimate2 = 1 / alpha2 * np.sqrt(numerical_integration_2D(np.square(alpha1 * Tq + alphaTV * div_p), h, h, N1, N2, rule))#1/alpha2 Only true for T=T_denoising!!!
    estimate = (estimate1 + estimate2)
    return estimate, estimate1, estimate2

def step_function_1D(x, value=1.0):
    return (np.where(x > 1, value, 0.0).astype(np.float32))

def solution_function_1D(x, a, b, alpha1, alpha2, value=1.0):
    return (np.where(x > 1, value - (1 - alpha1 * b) / (2 * alpha2 * b), (1 - alpha1 * a) / (2 * alpha2 * a)).astype(np.float32))

def T_blurring(A_vec, parameter):
    blur_para = parameter['blur_para']
    N1 = parameter['height']
    N2 = parameter['width']
    sm_p = parameter['sm_p']
    A = tf.reshape(A_vec, [N1, N2])
    blurred_image = sf.smooth_filter(A, blur_para, sm_p)
    return tf.reshape(blurred_image, [N1 * N2, 1])

def create_mask(N1, N2):
    NoStripes = int(N1/10)
    mask = np.ones((N1, N2)).astype(np.float32)
    mask[:,NoStripes::NoStripes]=0
    mask_vec = np.reshape(mask, (N1 * N2, 1))
    return mask_vec

def T_inpainting(u_vec, parameter):
    mask = parameter
    return tf.multiply(mask, u_vec)

def T_denoising(u_vec, parameter):
    return u_vec




if __name__ == "__main__":
    d = params['d']
    image_nr = params['image_nr']
    
    if d == 1:
        pass
    elif d == 2:
        n= 2**6
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
            image = image_functions.input_image(image_nr)
        y = image.astype(np.float32)

        N1, N2 = y.shape
        # Set N1 and N2 in params
        params['N1'] = N1
        params['N2'] = N2
        alpha1 = params['alpha1']
        alpha2 = params['alpha2']
        alpha_TV = params['alpha_TV']
        
        problem = params['problem']
        if problem == 'deblurring':
            blur_para = params['blur_para']
            sm_p = params['sm_p']
            parameter = {'blur_para': blur_para, 'height': N1, 'width': N2, 'sm_p': sm_p}
            T = T_blurring
        elif problem == 'inpainting':
            mask = create_mask(N1, N2)
            parameter = mask
            T = T_inpainting
        elif problem == 'denoising':
            parameter = ()
            T = T_denoising

        y_vec = np.reshape(y, (N1 * N2, 1))
        g_vec = T(y_vec, parameter)
        g = np.reshape(g_vec, (N1, N2))

        noise = params['noise']
        path = f"{d}d/{problem}/image{image_nr}/{noise}"
        if noise == 'Gauss' or noise == 'Gauss_sp' or noise == 'Gauss_rv' or noise == 'Gauss_sp_rv':
            gnoise = params['gnoise']
            seed_nr = params['seed_nr']
            g = image_functions.add_gauss_noise(g, gnoise, seed_nr, c='real').astype(np.float32)
            path = f"{path}/gnoise{gnoise}"

        if noise == 'sp' or noise == 'Gauss_sp' or noise == 'sp_rv' or noise == 'Gauss_sp_rv':
            spnoise = params['spnoise']
            sp_ratio = params['sp_ratio']
            g = image_functions.add_salt_pepper_noise(g, spnoise, sp_ratio, seed_nr=0)
            path = f"{path}/spnoise{spnoise}"

        if noise == 'rv' or noise == 'sp_rv' or noise == 'Gauss_rv' or noise == 'Gauss_sp_rv':
            rvnoise = params['rvnoise']
            g = image_functions.add_random_impulse_noise(g, rvnoise, seed_nr=0)
            path = f"{path}/rvnoise{rvnoise}"

        hx = 1 / N1
        hy = 1 / N2
        params['h'] = hx  # Dynamically calculated based on image size
        params['hx'] = hx
        params['hy'] = hy
        tensor = tf.linspace(0.0, float(g.shape[0] - 1), g.shape[0]) / N1
        x_tf = tf.reshape(tensor, (-1, 1))
        y_tf = tf.reshape(tensor, (-1, 1))
        X, Y = tf.meshgrid(x_tf[:, 0], y_tf[:, 0])
        xy = tf.stack([tf.reshape(X, [-1]), tf.reshape(Y, [-1])], axis=1)
        g_vec = tf.reshape(g, [N1 * N2, 1])

        for i, (hidden_layers, activations, l2_reg_key, max_value_key, use_kernel_constraint_key, use_bias_constraint_key) in enumerate(params['configurations']):
            l2_reg = params[l2_reg_key]
            max_value = params[max_value_key]
            use_kernel_constraint = params[use_kernel_constraint_key]
            use_bias_constraint = params[use_bias_constraint_key]
            
            log_dir = params['log_dir']
            os.makedirs(log_dir, exist_ok=True)

            with DualLogger(log_dir):
                plt.figure()
                plt.axis('off')
                plt.imshow(y, cmap='gray', vmin=0, vmax=1)
                plt.title('Original Image')
                plt.savefig(f'{log_dir}/original_image.png')
                plt.show()
                plt.close()

                plt.axis('off')
                plt.imshow(g, cmap='gray')
                plt.title('Observation')
                plt.savefig(f'{log_dir}/observation.png')
                plt.show()
                plt.close()

                model = SimpleNN(
                    hidden_layers, 
                    activations, 
                    l2_reg, 
                    max_value,
                    use_kernel_constraint=use_kernel_constraint, 
                    use_bias_constraint=use_bias_constraint
                )
                optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
                test = model(xy)
                epochs = params['epochs']
                tv_type = params['tv_type']
                
                best_model, best_image, error_data = train(model, optimizer, xy, g_vec, params, T, parameter)


                save_parameters(log_dir, params)  # Save parameters in .txt, .tex, and .pkl formats

                best_model.set_weights(best_model.get_weights())
                y_pred = best_model(xy, training=False).numpy()

                plt.axis('off')
                plt.imshow(np.reshape(y_pred, (N1, N2)), cmap='gray', vmin=0, vmax=1)
                plt.title('Final Reconstruction')
                plt.savefig(f'{log_dir}/reconstruction.png')
                plt.close()

                plt.axis('off')
                plt.imshow(np.reshape(best_image, (N1, N2)), cmap='gray', vmin=0, vmax=1)
                plt.title('Best Training Image')
                plt.savefig(f'{log_dir}/best_training_image.png')
                plt.show()
                plt.close()
                print(image_functions.calculate_quality_measures(y, np.reshape(best_image, (N1, N2))))
                save_essential_data(log_dir=log_dir,params=params,best_image=best_image,error_data=error_data,g=g) #store also u(it is a projection of thr true solution on the grid)...also u-bestimage in a norm
            sys.stdout = original_stdout
