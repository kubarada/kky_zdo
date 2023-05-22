import cv2 as cv
import numpy as np
import os


directory = 'cvat_dataset/images/default/'

# iterate over files in
# that directory
i = 0
z = 0

files = []
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    files.append(f)

for filename in files:
    img = cv.imread(filename,cv.IMREAD_GRAYSCALE)
    h, w = img.shape
    img = cv.medianBlur(img, 5)
    th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, \
                               cv.THRESH_BINARY_INV, 11, 2)

    cols = th3.shape[1]
    horizontal_size = cols // 30
    horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv.erode(th3, horizontalStructure)
    horizontal = cv.dilate(horizontal, horizontalStructure)
    cv.imwrite('cvat_dataset/images/output_th/' + str(z) + '.png', horizontal)
    z = z + 1

    lines_list = []
    lines = cv.HoughLinesP(
        horizontal,  # Input edge image
        10,  # Distance resolution in pixels
        np.pi / 180,  # Angle resolution in radians
        threshold=150,  # Min number of votes for valid line
        minLineLength=w / 1.8,  # Min allowed length of line
        maxLineGap=w/15  # Max allowed gap between line for joining them
    )

    if lines is None:
        print('No lines to detect in file ' +str(filename))
        #del img, h, w, th3, kernel, morph, lines_list, lines
        continue


    # Iterate over points
    for points in lines:
        x1, y1, x2, y2 = points[0]
        cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        lines_list.append([(x1, y1), (x2, y2)])

    cv.imwrite('cvat_dataset/images/output/' + str(i) + '.png',img)
    i = i +1
    #del  h, w, th3, kernel, morph, lines_list, lines