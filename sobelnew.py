import cv2
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
IMG_PATH = './images/'
IMG_NAME = 'oppenheimer_1.png'
# Load the image in grayscale
# = cv2.imread(IMG_PATH + IMG_NAME, cv2.IMREAD_GRAYSCALE)
#
# No need to convert to grayscale again
#mage =b

def sobel_image(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)

# Sobel filter in the Y direction 
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# Calculate the magnitude of the gradient
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)

    return sobel_combined



def threshold_image(image, strong_threshold, weak_threshold):

    
    strong_threshold_value = strong_threshold * np.max(image)
    weak_threshold_value = weak_threshold * np.max(image)

    # Apply thresholding
    image[image < weak_threshold_value] = 0
    image[image >= strong_threshold_value] = np.max(image)

    # Find in-between values
    inbetween_values = np.logical_and(image >= weak_threshold_value, image < strong_threshold_value)
    inbetween_values = inbetween_values * image

    rows, cols = inbetween_values.shape

    #   Perform edge tracking by hysteresis
    for    i in range(1, rows - 1):
        for j in range(1, cols - 1):
         
            if inbetween_values[i, j] == 0:
                continue
            else:
                if (image[i, j + 1] >= inbetween_values[i, j] or
                    image[i, j - 1] >= inbetween_values[i, j] or
                    image[i + 1, j] >= inbetween_values[i, j] or
                    image[i - 1, j] >= inbetween_values[i, j] or
                    image[i + 1, j + 1] >= inbetween_values[i, j] or
                    image[i + 1, j - 1] >= inbetween_values[i, j] or
                    image[i - 1, j + 1] >= inbetween_values[i, j] or
                    image[i - 1, j - 1] >= inbetween_values[i, j]):
                    image[i, j] = np.max(image)
                else:
                    inbetween_values[i, j] = 0
    image[image <= strong_threshold_value] = 0
    return image
    



