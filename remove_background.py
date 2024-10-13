import numpy as np
import cv2
import os
from wavelets import dwt2_edge_detection
from gaussian_smoothing import gaussian_smooth


def remove_background(
    image_path,
    method="wave",
    convex_hull=False
):
    """
    Removes the background from an image using edge detection and contour analysis.

    Parameters:
        image_path (str): Path to the input image.
        output_mask_path (str): Path to save the mask image.
        output_image_path (str): Path to save the output image with background removed.
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
    # Here we do edge detection

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
    # We treat edges and make *sure* that the contour is closed

    # Define the threshold values for mask (mostly for sobel implementatoin)
    lower_threshold = 20
    upper_threshold = 255

    binary_mask = cv2.inRange(edges, lower_threshold, upper_threshold)

    cv2.imwrite('output/edges.png', binary_mask)
    cv2.imshow('Edges', binary_mask)

    kernel = np.ones((22, 22), np.uint8) # kernal size to close gaps
    dilated = cv2.dilate(binary_mask, kernel, iterations=3)
    eroded = cv2.erode(dilated, kernel, iterations=2)

    cleaned = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('output/cleaned.png', cleaned)

#####################################################
    # Here we assume that the contur is connected. And we do flood fill from each corner
    # This will give us where "holes" inside the contour are. We use this to fill the mask

    inverted_mask = cv2.bitwise_not(cleaned)

    flood_filled = inverted_mask.copy()
    h, w = inverted_mask.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)  # Padding

    # flood fill from each of the corners
    corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    for corner in corners:
        cv2.floodFill(flood_filled, mask, corner, 0)

    flood_filled_inv = cv2.bitwise_not(flood_filled)

    filled_person = np.where(flood_filled_inv == 0, 255, cleaned)

    cv2.imwrite('output/filled_person.png', filled_person)
#####################################################
    # For our convex hull solution we find the biggest contour.
    # This filled contour is also used for just mask solution.

    contours, _ = cv2.findContours(filled_person, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found")
        return

    mask = np.zeros_like(image_gray)
    largest_contour = max(contours, key=cv2.contourArea)
    image_with_contour = image.copy()
    cv2.drawContours(image_with_contour, [largest_contour], -1, (0, 255, 0), thickness=3)
    cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
    cv2.imwrite('output/after_dilated_mask.png', mask)

    if convex_hull:
        convex_hull = cv2.convexHull(largest_contour)
        cv2.drawContours(mask, [convex_hull], -1, (255), thickness=cv2.FILLED)

        # Refine mask using morphological operations (optional, for smoothing)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
    else:
        # Make mask smoother
        smooth_kernel = np.ones((50, 50), np.uint8)
        cleaned_smooth = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, smooth_kernel)

        # Gaussian Blur for softer edges
        smoothed_mask = cv2.GaussianBlur(cleaned_smooth, (15, 15), 0)

        # Convert back to a binary mask using thresholding
        _, mask = cv2.threshold(smoothed_mask, 127, 255, cv2.THRESH_BINARY)


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
remove_background('./images/girl.jpeg', method="wave", convex_hull=False)

