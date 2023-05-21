import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

PATH_TO_IMG = 'cvat_dataset/images/default/SA_20211104-131923_incision_crop_0.jpg'

img = cv.imread(PATH_TO_IMG, cv.IMREAD_GRAYSCALE)
h,w = img.shape
assert img is not None, "file could not be read, check with os.path.exists()"
img = cv.medianBlur(img,5)
ret,th1 = cv.threshold(img,110,255,cv.THRESH_BINARY)
cv.imwrite('tresholded1.png', th1)
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)
cv.imwrite('tresholded2.png', th2)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY_INV,11,2)
cv.imwrite('tresholded3.png', th3)

kernel = np.ones((1,3),np.uint8)
morph = cv.morphologyEx(th3, cv.MORPH_OPEN, kernel)
kernel = np.ones((1,3),np.uint8)
morph = cv.erode(morph,kernel,iterations = 3)
kernel = np.ones((1,5),np.uint8)
morph = cv.dilate(morph,kernel,iterations = 1)


cv.imwrite('tresholded.png', th3)

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

cv.imwrite('detectedLinesTresh.png', img)

vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 5))
horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 1))

# Extract vertical lines
vertical_lines = cv.erode(th3, vertical_kernel, iterations=1)
vertical_lines = cv.dilate(vertical_lines, vertical_kernel, iterations=2)


sobel_y = cv.Sobel(img, cv.CV_64F, 2, 0, ksize=3)

# Normalizace výsledku na rozsah 0-255
sobel_y = cv.normalize(sobel_y, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
cv.imwrite('sobelLines.png', sobel_y)


# horizontal_lines = cv.erode(th3, horizontal_kernel, iterations=1)
# horizontal_lines = cv.dilate(horizontal_lines, horizontal_kernel, iterations=2)
# horizontal_lines = cv.bitwise_not(horizontal_lines)
# #
# # # Combine the vertical and minimized horizontal lines
# vertical_lines = cv.bitwise_or(vertical_lines, horizontal_lines)
cv.imwrite('verticalTresh.png', vertical_lines)

lines1 = cv.HoughLinesP(
    vertical_lines,  # Input edge image
    10,  # Distance resolution in pixels
    np.pi / 180,  # Angle resolution in radians
    threshold=None,  # Min number of votes for valid line
    minLineLength=h/2,  # Min allowed length of line
    maxLineGap=10 # Max allowed gap between line for joining them
)
print(lines1)

theta_min = 0 * np.pi / 180  # 45 degrees
theta_max = 180 * np.pi / 180  # 135 degrees

# Iterate over points

# Filter and draw detected vertical lines on the image
if lines1 is not None:
    for line in lines1:
        x1, y1, x2, y2 = line[0]
        theta = np.arctan2(y2 - y1, x2 - x1)  # Calculate the angle of the line
        if theta_min < theta < theta_max:  # Filter lines with angle between 45 and 135 degrees
            cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)


cv.imwrite('detectedLinesVertical.png', img)