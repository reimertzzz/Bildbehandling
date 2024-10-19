import cv2
import numpy as np
from scipy.signal import convolve2d
from utils import show_image

def sobel_edge_detection(image):
    # Perform edge detection using Sobel

    # Define Sobel filters
    sobel_horizontal = np.array([[1, 2, 1],
                                 [0, 0, 0],
                                 [-1, -2, -1]], dtype=np.float32)

    sobel_vertical = np.array([[1, 0, -1],
                               [2, 0, -2],
                               [1, 0, -1]], dtype=np.float32)
    
    # Apply Sobel filters to get horizontal and vertical gradients
    image_horizontal = cv2.filter2D(image.astype(np.float32), -1, sobel_horizontal)
    image_vertical = cv2.filter2D(image.astype(np.float32), -1, sobel_vertical)

    # Compute gradient magnitude approximation
    image_approximation = np.abs(image_horizontal) + np.abs(image_vertical)

    return image_approximation


def edge_enhancer(edges_image):
    edges = edges_image.copy()
    # Perform edge enhancer on an edge mask
    
    # Set thresholds
    strong_threshold = 0.6
    weak_threshold = 0.3

    # Calculate threshold values
    strong_threshold_value = strong_threshold * np.max(edges)
    weak_threshold_value = weak_threshold * np.max(edges)

    # Apply thresholding
    edges[edges < weak_threshold_value] = 0
    edges[edges >= strong_threshold_value] = np.max(edges)

    # Find in-between values
    inbetween_values = np.logical_and(edges >= weak_threshold_value, edges < strong_threshold_value)
    inbetween_values = inbetween_values * edges

    # Get image dimensions
    rows, cols = inbetween_values.shape

    # Perform edge tracking by hysteresis
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if inbetween_values[i, j] == 0:
                continue
            else:
                if (edges[i, j + 1] >= inbetween_values[i, j] or
                    edges[i, j - 1] >= inbetween_values[i, j] or
                    edges[i + 1, j] >= inbetween_values[i, j] or
                    edges[i - 1, j] >= inbetween_values[i, j] or
                    edges[i + 1, j + 1] >= inbetween_values[i, j] or
                    edges[i + 1, j - 1] >= inbetween_values[i, j] or
                    edges[i - 1, j + 1] >= inbetween_values[i, j] or
                    edges[i - 1, j - 1] >= inbetween_values[i, j]):
                    edges[i, j] = np.max(edges)
                else:
                    inbetween_values[i, j] = 0
    
    return edges