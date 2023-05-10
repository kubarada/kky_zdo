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
    edges = cv2.Canny(gray, 10, 150, apertureSize=3)

    kernel = np.ones((1, 3), np.uint8)
    edges_morph = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((1, 3), np.uint8)
    edges_morph = cv2.morphologyEx(edges_morph, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((1, 2), np.uint8)
    edges_morph = cv2.erode(edges_morph, kernel, iterations=1)
    edges_morph = cv2.dilate(edges_morph, kernel, iterations=5)
    # Apply HoughLinesP method to
    # to directly obtain line end points
    lines_list = []
    lines = cv2.HoughLinesP(
        edges_morph,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 180,  # Angle resolution in radians
        threshold=20,  # Min number of votes for valid line
        minLineLength=50,  # Min allowed length of line
        maxLineGap=250  # Max allowed gap between line for joining them
    )
    print(lines)

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