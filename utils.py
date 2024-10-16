import pywt
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, io, data, morphology, transform
import cv2
import math
from collections.abc import Iterable

def display_wavelet_coefficients(approximation, details, caption=None):
    # Display on a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    titles = ['Approximation', 'Horizontal Details', 'Vertical Details', 'Diagonal Details']
    # Display approximation
    coeffs = [approximation] + list(details)
    for i, a in enumerate(coeffs):
        row = i // 2; col = i % 2
        axes[row, col].imshow(a, cmap='gray')
        axes[row, col].set_title(titles[i])
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])

    if caption:
        fig.suptitle(caption)
    plt.tight_layout()
    plt.show()


def show_image(image, caption=None):
    # Have to explicitly tell matplotlib it's grayscale
    if len(image.shape) == 2: 
        plt.imshow(image, cmap='gray')
    else: 
        plt.imshow(image)
    if caption: plt.title(caption)
    plt.axis('off')
    plt.show()

def show_images(images, captions=None):
    # Handle single image
    if not isinstance(images, Iterable):
        show_image(images, captions)
        return
    color_type = [None] * len(images)
    for i, image in enumerate(images):
        # Have to explicitly tell matplotlib it's grayscale
        if len(image.shape) == 2: 
            color_type[i] = 'gray'
            # plt.imshow(image, cmap='gray')
    # Get size of grid to accomodate all images
    n_cols = math.ceil(math.sqrt(len(images)))
    n_rows = math.ceil(len(images) / n_cols)
    # Plot images
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 8))
    for i, img in enumerate(images):
        row = i // n_rows; col = i % n_cols
        if n_rows > 1:
            ax = axes[row, col]
        else:
            ax = axes[row]
        ax.imshow(img, color_type[i])
        ax.set_xticks([])
        ax.set_yticks([])
        if captions:
            ax.set_title(captions[i])
    plt.show()

def dwt2_gray(image, wavelet, level=1):
    ''' Perform dwt2 and transform each image to the original size. '''
    orig_size = image.shape
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)
    # Transform to original size
    coeffs[0] = transform.resize(coeffs[0], orig_size)
    for level, details in enumerate(coeffs[1:]):
        details = tuple([transform.resize(d, orig_size) for d in details])
        coeffs[level+1] = details
    return coeffs

def dwt2_color(image):
    # Perform dwt2 on a color image by splitting it into colors,
    # performing dwt2, and finally stack together
    
    # Split into color channels
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Perform dwt2
    coeffs2_R = pywt.dwt2(R, 'haar')
    coeffs2_G = pywt.dwt2(G, 'haar')
    coeffs2_B = pywt.dwt2(B, 'haar')

    # Get coefficients
    ax_R, (hx_R, vx_R, dx_R) = coeffs2_R
    ax_G, (hx_G, vx_G, dx_G) = coeffs2_G
    ax_B, (hx_B, vx_B, dx_B) = coeffs2_B

    # Stack coefficients
    ax = np.stack((ax_R, ax_G, ax_B), axis=-1)
    hx = np.stack((hx_R, hx_G, hx_B), axis=-1)
    vx = np.stack((vx_R, vx_G, vx_B), axis=-1)
    dx = np.stack((dx_R, dx_G, dx_B), axis=-1)

    coeffs = [ax, hx, vx, dx]

    return coeffs

def load_image(filename: str) -> np.ndarray:
    return io.imread(filename)

def is_grayscale(image: np.ndarray) -> bool:
    ''' Checks if image is gray by checking number of dimensions. '''
    return len(image.shape) == 2