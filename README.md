
# DeepTV Image Processing Project

## Overview

This code is based on the paper 

[1] Andreas Langer and Sara Behnamian. 'DeepTV: A deep neural network approach for total variation minimization'. In preparation, 2024.
                                             
Read and use at your own risk.

This repository contains the implementation of a neural network-based image processing system, DeepTV, that supports various image restoration tasks such as denoising, deblurring, and inpainting. It leverages custom-defined image processing and filtering algorithms along with a TensorFlow neural network to enhance and restore images corrupted by noise, blurring, or missing data. 

If you use the code, please make sure to cite [1]. 

### Contents

The repository consists of the following Python scripts:

- `NN_image_processing.py`: Main script that defines and trains the neural network, applying various image processing techniques.
- `smooth_filter.py`: Implements smoothing filters such as Gaussian and mean filters.
- `image_functions.py`: Contains utility functions for image loading, adding noise, and calculating quality measures like SSIM and PSNR.
- `main1d.py`: Script for denoising 1D signals using neural networks and energy minimization techniques via the `NN_image_processing` module.
- `main2d.py`: Script for 2D image restoration tasks (denoising, inpainting, deblurring) using neural networks and total variation regularization techniques via the `NN_image_processing` module.
- `Images/`: Folder where input images are stored.


## Features

- **Denoising**: Removes Gaussian, salt-and-pepper, and random-valued impulse noise from images.
- **Deblurring**: Recovers sharp images from blurred ones using filters and neural networks.
- **Inpainting**: Fills in missing pixels in images based on surrounding data using deep learning techniques.
- **Custom Loss Functions**: Utilizes L2 and L1 loss terms, along with TV (Total Variation) regularization for improved image restoration.
- **Custom Neural Network Architecture**: Configurable layers, activation functions, and regularization settings for various restoration problems.
- **Logging and Error Estimation**: Logs intermediate and final results and computes error estimates during training.
  
## Setup

### Prerequisites

Make sure you have Python 3.x installed and the following Python libraries:

- `tensorflow==2.16.1`
- `numpy==1.26.4`
- `matplotlib==3.9.0`
- `scikit-image==0.24.0`
- `scipy==1.13.1`

To install the dependencies, run:

```bash
pip install -r requirements.txt
```

### Folder Structure

- **`NN_image_processing.py`**: Main script to execute different image processing tasks.
- **`smooth_filter.py`**: Contains custom filter implementations used for blurring and smoothing images.
- **`image_functions.py`**: Includes helper functions for adding noise, loading images, and calculating image quality metrics.
- **`Images/`**: Folder for storing input images used for training and testing the model.
- **`main1d.py`**: Executes 1D signal denoising using neural networks and energy minimization methods, leveraging `NN_image_processing`.
- **`main2d.py`**: Handles 2D image restoration tasks (denoising, inpainting, deblurring) using neural networks and total variation regularization via `NN_image_processing`.


### Running the Code

To run the image processing script, execute:

```bash
python NN_image_processing.py
```
Note: For larger images that require more computational resources, it's recommended to submit the job using SLURM.

This script will train a neural network on a specified image, apply noise or blurring, and then attempt to restore the image.

### Configuration

You can configure the task parameters by modifying the `params` dictionary in `NN_image_processing.py`. Below are some key parameters:
- `problem`: Choose between `'denoising'`, `'deblurring'`, or `'inpainting'`.
- `epochs`: Number of training epochs.
- `learning_rate`: Learning rate for the optimizer.
- `blur_para`: Parameters for the blurring kernel.
- `noise`: Type of noise to add (`'Gauss'`, `'sp'`, `'rv'`, etc.).
- `alpha1`, `alpha2`, `alpha_TV`: Regularization parameters for the loss function.

### Logging and Visualization

Training progress, losses, and reconstructed images are logged in the specified `log_dir`. The logs include:
- **Reconstructed Images**: Best results at various intervals.
- **Error Plots**: Comparison of real error and error estimates.
- **Quality Metrics**: PSNR, SSIM, and MAE between the original and reconstructed images.

## Results

The model generates intermediate and final results showing the effectiveness of the applied image restoration techniques. These outputs are saved as images and logs within the specified `log_dir`.

## Example

Below is a simple usage example for running denoising on an image:

```python
params = {
    'problem': 'denoising',           # Task type: 'denoising', 'inpainting', or 'deblurring'
    'd': 2,                           # Dimension of the task: 1 or 2
    'image_nr': 0,                    # Image number or custom image input
    'height': 96,                     # Image height (only for 'image_nr'=0)
    'width': 96,                      # Image width (only for 'image_nr'=0)
    'epochs': 10001,                  # Number of training epochs
    'learning_rate': 1e-3,            # Learning rate for the optimizer
    'noise': 'Gauss_sp',              # Type of noise to be added (e.g., 'Gauss', 'sp', 'rv')
    'gnoise': (0.0, 0.05),            # Parameters for Gaussian noise (mean, standard deviation)
    'alpha1': 1.0,                    # Regularization parameter for L1 norm
    'alpha2': 7.0,                    # Regularization parameter for L2 norm
    'alpha_TV': 1.0,                  # Regularization parameter for Total Variation (TV)
    'tv_type': 'TV21',                # Type of TV regularization ('TV2', 'TV21' for 2D)
    'log_dir': 'logs',                # Directory to save training logs and outputs
    'save_plots': True,               # Whether to save plots during training
    'plot_interval': 100,             # Interval at which plots are saved
    'configurations': [
        ([128, 128, 128], ['relu', 'relu', 'relu'], 'l2_reg', 'max_value', True, True)
    ],                                # Neural network configuration (layers, activations, etc.)
    ...
}
```
This is a setup for denoising a 2D image. The model will be trained for denoising a given image and save the restored image along with the error metrics. 

You can further customize it by modifying parameters such as noise type, regularization terms, image size, and neural network architecture based on your specific task.


## License

This project is licensed under the MIT License.

## Author

- Andreas Langer - Sara Behnamian

---

This project serves as a foundation for image processing and can be further expanded with more sophisticated models, loss functions, and filtering techniques. If you have any questions, feel free to open an issue on GitHub, or email us: andreas.langer@math.lth.se, sara.behnamian@sund.ku.dk.

