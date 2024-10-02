import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
import matplotlib.pyplot as plt


def gaussian_smooth(img_path):
    img_gray = Image.open(img_path).convert('L')
    np_image = np.asarray(img_gray)
    std_deviation = np.std(img_gray)
    smoothed_img = gaussian_filter(np_image, std_deviation)
    return smoothed_img

def display_smooth_img(img_path):
    img = gaussian_smooth(img_path)
    plt.imshow(img, cmap='grey')
    plt.axis('on')
    plt.title('Gaussian smoothed image')
    plt.show()
