import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
import matplotlib.pyplot as plt


def gaussian_smooth(img_path, sigma):
    img_gray = Image.open(img_path).convert('L')
    np_image = np.asarray(img_gray)
    smoothed_img = gaussian_filter(np_image, sigma)
    return smoothed_img

def display_smooth_img(img_path, sigma=8.5):
    img = gaussian_smooth(img_path, sigma)
    plt.imshow(img, cmap='grey')
    plt.axis('on')
    plt.title('Gaussian smoothed image')
    plt.show()
