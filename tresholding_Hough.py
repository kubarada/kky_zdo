import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

PATH_TO_IMG = 'cvat_dataset/images/default/SA_20211012-164802_incision_crop_0.jpg'

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
th3 = cv.morphologyEx(th3, cv.MORPH_OPEN, kernel)
kernel = np.ones((1,3),np.uint8)
th3 = cv.erode(th3,kernel,iterations = 3)
kernel = np.ones((1,5),np.uint8)
th3 = cv.dilate(th3,kernel,iterations = 1)


cv.imwrite('tresholded.png', th3)

lines_list = []
lines = cv.HoughLinesP(
    th3,  # Input edge image
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