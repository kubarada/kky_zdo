import cv2
import numpy as np
import os


# Read image
PATH_TO_IMG = 'cvat_dataset/images/default/SA_20211012-164802_incision_crop_0.jpg'
IMG_TITLE = 'Incision/Stitch image'

image = cv2.imread(PATH_TO_IMG)

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite('rgb2gray.png', gray)



# Use canny edge detection
edges = cv2.Canny(gray,60,120,apertureSize=3)
kernel = np.ones((1,2),np.uint8)
edges_morph = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
kernel = np.ones((1,5),np.uint8)
edges_morph = cv2.morphologyEx(edges_morph, cv2.MORPH_OPEN, kernel)

edges_morph = cv2.dilate(edges_morph,kernel,iterations = 6)

kernel = np.ones((2,1),np.uint8)
edges_morph = cv2.erode(edges_morph,kernel,iterations = 1)

cv2.imwrite('cannyFilter.png', edges_morph)


# Apply HoughLinesP method to
# to directly obtain line end points
lines_list = []
lines = cv2.HoughLinesP(
    edges,  # Input edge image
    20,  # Distance resolution in pixels
    np.pi / 170,  # Angle resolution in radians
    threshold=80,  # Min number of votes for valid line
    minLineLength=90,  # Min allowed length of line
    maxLineGap=18 # Max allowed gap between line for joining them
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
cv2.imwrite('detectedLines.png', image)
