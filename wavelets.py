import pywt
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, io, data, morphology, transform
import cv2

def dwt2_edge_detection(
    image: np.ndarray,
    wavelet: str,
    depth: int,
    threshold_upper: int,
    threshold_lower: int,
    weights = (1, 1, 1)
) -> np.ndarray:
    """ 
    Edge detection for a gray image.
    Parameters: 
        image (np.ndarray): the gray image as a numpy array.
        wavelet (str): the type of wavelet used. See pywt.wavedec2 for alternatives.
        depth (int): the level of depth of the wavelet transform
        threshold_upper (int): the upper limit for what is not edges [0,255]
        threshold_lower (int): the lower limit for what is not edges [0,255]
        weights (tuple): The weight given to horizontal, vertical and diagonal details, respectively
    returns: an numpy array. 
    """
    # Perform wavelet transformation to a depth of 'depth'
    coeffs = dwt2_gray(image, wavelet, depth)

    # Get approximation and deepest level of approximation
    approximation = coeffs[0]
    details = coeffs[1]

    # Get edge mask by using all three details
    edges = details[0] * weights[0] + details[1] * weights[1] + details[2] * weights[2]
    
    # Convert edges to [0, 255] and uint8
    edges /= np.max(edges)
    edges *= 255
    edges = edges.astype(np.uint8)

    # Get mask by taking strong or small edges
    mask = 1 - ((edges > threshold_upper) | (edges < threshold_lower))
    return mask.astype(np.uint8) * 255


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


def get_outline(mask: np.ndarray) -> np.ndarray:
    """ Get outline by choosing leftmost and rightmost pixel in each row. """
    start_edge = [np.where(row)[0][0] if np.any(row) else 0 for row in mask]
    end_edge = [np.where(row)[0][-1] if np.any(row) else 0 for row in mask]

    obj_mask = np.zeros_like(mask, dtype=bool)
    for i, (start, end) in enumerate(zip(start_edge, end_edge)):
        if start == end: continue
        obj_mask[i, start:end+1] = True
    
    return (obj_mask).astype(np.uint8) * 255

