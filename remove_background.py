import numpy as np
import cv2
import os
from wavelets import dwt2_edge_detection
from gaussian_smoothing import gaussian_smooth
from scipy.ndimage import label
from skimage.measure import regionprops
# from matplotlib.pyplot import plt

def remove_background(
    image_path,
    method="sobel",
    blur=11,
    canny_low=15,
    canny_high=255,
    min_area_ratio=0.005,
    max_area_ratio=0.95,
    mask_dilate_iter=15,
    mask_erode_iter=15,
    mask_color=(0.0, 0.0, 0.0)
):
    """
    Removes the background from an image using edge detection and contour analysis.

    Parameters:
        image_path (str): Path to the input image.
        output_mask_path (str): Path to save the mask image.
        output_image_path (str): Path to save the output image with background removed.
        blur (int): Gaussian blur kernel size (should be odd).
        canny_low (int): Lower threshold for the Canny edge detector.
        canny_high (int): Upper threshold for the Canny edge detector.
        min_area_ratio (float): Minimum area ratio of contours to be considered.
        max_area_ratio (float): Maximum area ratio of contours to be considered.
        mask_dilate_iter (int): Number of iterations for mask dilation.
        mask_erode_iter (int): Number of iterations for mask erosion.
        mask_color (tuple): Color to use for the background (default is black).
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Convert image to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_smooth = gaussian_smooth(image_gray,2)


########################################################
    # Define Sobel filters
    if method == "sobel":

        sobel_horizontal = np.array([[1, 2, 1],
                                    [0, 0, 0],
                                    [-1, -2, -1]])

        sobel_vertical = np.array([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]])

        # Apply horizontal and vertical Sobel filters using cv2.filter2D
        obama_horizontal = cv2.filter2D(image_smooth, -1, sobel_horizontal)
        obama_vertical = cv2.filter2D(image_smooth, -1, sobel_vertical)
        edges = np.abs(obama_vertical) + np.abs(obama_horizontal)
        cv2.imwrite('output/sobel_raw.png', edges)

    else:
        # Apply Canny Edge Detection
        edges = dwt2_edge_detection(image_gray, 'haar', 2, 230, 17) # Custom edge detection
        cv2.imwrite('output/wavelet_edges.png', edges)

########################################################

    # Define the threshold values
    lower_threshold = 20
    upper_threshold = 255

    binary_mask = cv2.inRange(edges, lower_threshold, upper_threshold)

    cv2.imwrite('output/edges.png', binary_mask)
    cv2.imshow('Edges', binary_mask)

    kernel = np.ones((10, 10), np.uint8)
    dilated = cv2.dilate(binary_mask, kernel, iterations=3)
    cv2.imwrite('output/dilated.png', dilated)

    cleaned = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('output/cleaned.png', cleaned)

#####################################################
    # Step 6: Find contours and select the largest one (assuming it is the person)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found")
        return

    # Step 7: Create a blank mask and draw the largest contour
    mask = np.zeros_like(image_gray)
    largest_contour = max(contours, key=cv2.contourArea)
    image_with_contour = image.copy()
    cv2.drawContours(image_with_contour, [largest_contour], -1, (0, 255, 0), thickness=3)
    cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
    cv2.imwrite('output/after_dilated_mask.png', mask)

    mask = np.zeros_like(image_gray)
    convex_hull = cv2.convexHull(largest_contour)
    cv2.drawContours(mask, [convex_hull], -1, (255), thickness=cv2.FILLED)

     # Refine mask using morphological operations (optional, for smoothing)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Apply the mask to the original image to remove the background
    result = cv2.bitwise_and(image, image, mask=mask)

    # Step 9: Display or save the final mask
    cv2.imwrite('output/output_mask.png', result)
    cv2.imshow('result mask', result)
    cv2.imshow('contour', image_with_contour)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# countour_output.png
# girl.jpeg
# plain_background_portrait.jpg
# girl_with_sharp_background.jpg
# oppenheimer_1.png
# lena.jpeg
remove_background('./images/girl.jpeg', method="sobel")
