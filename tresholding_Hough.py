import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

PATH_TO_IMG = 'cvat_dataset/images/default/SA_20211012-181437_incision_crop_0.jpg'

img = cv.imread(PATH_TO_IMG, cv.IMREAD_GRAYSCALE)
h,w = img.shape
assert img is not None, "file could not be read, check with os.path.exists()"
img = cv.medianBlur(img,5)
ret,th1 = cv.threshold(img,110,255,cv.THRESH_BINARY)
cv.imwrite('failed/tresholded1.png', th1)
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)
cv.imwrite('failed/tresholded2.png', th2)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY_INV,11,2)
cv.imwrite('failed/tresholded3.png', th3)

kernel = np.ones((1,3),np.uint8)
morph = cv.morphologyEx(th3, cv.MORPH_OPEN, kernel)
kernel = np.ones((1,3),np.uint8)
morph = cv.erode(morph,kernel,iterations = 3)
kernel = np.ones((1,5),np.uint8)
morph = cv.dilate(morph,kernel,iterations = 1)


cv.imwrite('failed/tresholded.png', th3)

lines_list = []
lines = cv.HoughLinesP(
    morph,  # Input edge image
    10,  # Distance resolution in pixels
    np.pi / 180,  # Angle resolution in radians
    threshold=150,  # Min number of votes for valid line
    minLineLength=w/1.8,  # Min allowed length of line
    maxLineGap=10 # Max allowed gap between line for joining them
)
print(lines)

# Iterate over points
for points in lines:
    # Extracted points nested in the list
    x1, y1, x2, y2 = points[0]
    # Draw the lines joing the points
    # On the original image
    cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Maintain a simples lookup list for points
    lines_list.append([(x1, y1), (x2, y2)])

cv.imwrite('failed/detectedLinesTresh.png', img)
thresh = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 17, 2)

# Determine the mean value of the binary image
mean_value = np.mean(thresh)

# Set Canny parameters based on the mean value
low_threshold = int(mean_value * 0.5)
high_threshold = int(mean_value * 1)

# Apply Canny edge detection
edges = cv.Canny(thresh, low_threshold, high_threshold)
cv.imwrite('canny.png', edges)
kernel_vertical = cv.getStructuringElement(cv.MORPH_RECT, (1, 6))
vertical_lines = cv.morphologyEx(edges, cv.MORPH_OPEN, kernel_vertical)
vertical_lines = cv.dilate(vertical_lines, kernel_vertical, iterations=5)
kernel_horizontal = cv.getStructuringElement(cv.MORPH_RECT, (5, 1))
horizontal_lines = cv.morphologyEx(edges, cv.MORPH_OPEN, kernel_horizontal)
enhanced_lines = cv.bitwise_or(vertical_lines, horizontal_lines)

cv.imwrite('verticalTresh.png', vertical_lines)

lines1 = cv.HoughLinesP(
    vertical_lines,  # Input edge image
    1,  # Distance resolution in pixels
    np.pi / 180,  # Angle resolution in radians
    threshold=None,  # Min number of votes for valid line
    minLineLength=1,  # Min allowed length of line
    maxLineGap=25 # Max allowed gap between line for joining them
)
print(lines1)

from detector import keypoints_postprocessing, coordinates_control, detect_stitches
false1= 0
fin = detect_stitches(PATH_TO_IMG, false1)
fin = keypoints_postprocessing(lines1, img, 'stitch', PATH_TO_IMG)
print(fin)
fin = coordinates_control(fin, img, PATH_TO_IMG)
for points in fin:

    # Extracted points nested in the list
    x1, y1, x2, y2 = points[0][0]

    # Draw the lines joing the points
    # On the original image
    cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
cv.imwrite('detectedLinesVertical.png', img)