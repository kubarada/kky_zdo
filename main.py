import cv2
import numpy as np

# Read image
PATH_TO_IMG = 'cvat_dataset/images/default/SA_20211012-164802_incision_crop_0.jpg'
IMG_TITLE = 'Incision/Stitch image'

image = cv2.imread(PATH_TO_IMG)

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(gray)
# Use canny edge detection
edges = cv2.Canny(gray,50,100,apertureSize=3)
print(edges)
# Apply HoughLinesP method to
# to directly obtain line end points
lines_list = []
lines = cv2.HoughLinesP(
    edges,  # Input edge image
    5,  # Distance resolution in pixels
    np.pi / 180,  # Angle resolution in radians
    threshold=50,  # Min number of votes for valid line
    minLineLength=10,  # Min allowed length of line
    maxLineGap=4  # Max allowed gap between line for joining them
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
