import numpy as np
import cv2 as cv
from skimage.transform import probabilistic_hough_line

def horizontal_line_detection(PATH_TO_FILE):
    img = cv.imread(PATH_TO_FILE)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h, w = gray.shape
    # resizing, bcs works better on bigger pictures
    if w < 250:
        w = int(w * 2)
        h = int(h * 2)
        gray = cv.resize(gray, (w, h), interpolation=cv.INTER_CUBIC)
    th = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 17, 2)
    lines = cv.HoughLinesP(th, 1, np.pi / 180, 150, w * 0.75, w*0.1)
    lines_list = []
    if lines is not None:
        for line in lines:
            lines_list.append(line)
    else:
        w = int(w * 2)
        h = int(h * 2)
        th = cv.resize(th, (w, h), interpolation=cv.INTER_CUBIC)
        lines = cv.HoughLinesP(th, 1, np.pi / 180, 150, w * 0.75, w * 0.1)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi) < 30:  # adding lines with just certain angle
                    lines_list.append(line)
        else:
            print('No incision to detect in image ' + str(PATH_TO_FILE))

    if img.shape == gray.shape:
        return lines_list
    else:
        ratio = int(gray.shape[0] / img.shape[0])  # ratio between two images
        for i in range(0, len(lines_list)):
            lines_list[i][0] = (lines_list[i][0] / ratio).astype(np.int32)
        return lines_list


def vertical_line_detection(PATH_TO_FILE):
    img = cv.imread(PATH_TO_FILE)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (17, 17), 0)

    h, w = gray.shape
    w = int(w * 2)
    h = int(h * 2)
    gray = cv.resize(gray, (w, h), interpolation=cv.INTER_CUBIC)

    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 17, 2)

    kernel_vertical = cv.getStructuringElement(cv.MORPH_RECT, (1,8))
    eroded = cv.erode(thresh, kernel_vertical, iterations=1)
    thinner_line = cv.dilate(eroded, kernel_vertical, iterations=1)
    kernel_vertical = cv.getStructuringElement(cv.MORPH_RECT, (1, 9))
    thinner_line = cv.dilate(thinner_line, kernel_vertical, iterations=3)
    kernel_vertical = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))
    thinner_line = cv.erode(thinner_line, kernel_vertical, iterations=4)

    h, w = thinner_line.shape

    angles = np.linspace(-np.pi / 12, np.pi / 12, 100, endpoint=False)
    lines = probabilistic_hough_line(thinner_line, threshold=5, line_length=int(h*0.3), line_gap=int(h*0.1), theta=angles)
    stitches = []
    if lines is not None:
        for line in lines:
            start, end = line
            angle = np.arctan2(end[1] - start[1], end[0] - start[0]) * 180 / np.pi
            if np.abs(angle) > 50:
                stitches.append(line)
    else:
        print('No stitches to detect in image ', str(PATH_TO_FILE))

    stitches = np.array([[x1, y1, x2, y2] for (x1, y1), (x2, y2) in stitches])

    if img.shape == thresh.shape:
        return stitches
    else:
        ratio = int(thresh.shape[0] / img.shape[0])  # ratio between two images
        stitches = np.array(stitches) / ratio
        stitches = np.array(stitches, dtype=np.int32)
        return stitches

