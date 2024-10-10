import numpy as np
import cv2
import os
from wavelets import dwt2_edge_detection
from gaussian_smoothing import gaussian_smooth
from matplotlib.pyplot import plt

def remove_background(
    image_path,
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

    # Apply Canny Edge Detection
    edges = dwt2_edge_detection(image_gray, 'haar', 2, 255, 20) # Custom edge detection
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    # Find contours in edges
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Calculate contour areas and filter based on area thresholds
    contour_info = [(c, cv2.contourArea(c)) for c in contours]

    # Get the area of the image as a comparison
    image_area = image.shape[0] * image.shape[1]

    # Calculate max and min areas in terms of pixels
    max_area = max_area_ratio * image_area
    min_area = min_area_ratio * image_area

    # Create an empty mask
    mask = np.zeros(edges.shape, dtype=np.uint8)

    # Create a copy of the image to draw contours on
    image_with_contours = image.copy()

    # Prepare a list to store contours to draw
    contours_to_draw = []

    # Draw contours for all contours that are within the specified area range on the mask
    for contour, area in contour_info:
        if min_area < area < max_area:
            # Approximate contour to make it tighter without using convex hull
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(mask, [approx], -1, 255, thickness=cv2.FILLED)
            contours_to_draw.append(approx)

    # Draw the contours on the image_with_contours
    cv2.drawContours(image_with_contours, contours_to_draw, -1, (0, 255, 0), thickness=1)

    # Fill holes inside the mask by using flood fill from the background and then combining with the original mask
    flood_filled = mask.copy()
    h, w = mask.shape[:2]
    mask_flood_fill = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood_filled, mask_flood_fill, (0, 0), 255)
    flood_filled_inv = cv2.bitwise_not(flood_filled)
    mask = mask | flood_filled_inv

    # Smooth the mask to reduce noise and improve the result
    mask = cv2.dilate(mask, None, iterations=mask_dilate_iter)
    mask = cv2.erode(mask, None, iterations=mask_erode_iter)
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)

    # Create a 3-channel alpha mask
    mask_stack = mask.astype('float32') / 255.0
    mask_stack = np.dstack([mask_stack]*3)

    # Convert image to float for blending
    image_float = image.astype('float32') / 255.0

    # Blend the masked image onto a background
    masked = (mask_stack * image_float) + ((1 - mask_stack) * mask_color)
    masked = (masked * 255).astype('uint8')

    # Save the outputs
    return mask, masked, image_with_contours

# countour_output.png
# girl.jpeg
# plain_background_portrait.jpg
# girl_with_sharp_background.jpg
# oppenheimer_1.png
mask, masked_image, image_with_contours = remove_background('./images/girl.jpeg')
cv2.imwrite('output/mask.jpg', mask)
cv2.imwrite('output/masked_image.jpg', masked_image)
cv2.imwrite('output/contours.jpg', image_with_contours)
