import cv2
import numpy as np
import os


directory = 'cvat_dataset/images/default/'

# iterate over files in
# that directory
i = 0
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    image = cv2.imread(f)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use canny edge detection
    edges = cv2.Canny(gray, 50, 100, apertureSize=3)
    kernel = np.ones((1, 2), np.uint8)
    edges_morph = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((1, 6), np.uint8)
    edges_morph = cv2.morphologyEx(edges_morph, cv2.MORPH_OPEN, kernel)

    edges_morph = cv2.dilate(edges_morph, kernel, iterations=4)

    # Apply HoughLinesP method to
    # to directly obtain line end points
    lines_list = []
    lines = cv2.HoughLinesP(
        edges,  # Input edge image
        20,  # Distance resolution in pixels
        np.pi / 180,  # Angle resolution in radians
        threshold=70,  # Min number of votes for valid line
        minLineLength=70,  # Min allowed length of line
        maxLineGap=20  # Max allowed gap between line for joining them
    )
    if lines is None:
        continue
    else:
        # Iterate over points
        for points in lines:
            # Extracted points nested in the list
            x1, y1, x2, y2 = points[0]
            # Draw the lines joing the points
            # On the original image
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Maintain a simples lookup list for points
            lines_list.append([(x1, y1), (x2, y2)])

        # Save the result image
        cv2.imwrite('cvat_dataset/images/output/' + str(i) + '.png', image)
        i =+1