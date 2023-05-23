import os
from detector import horizontal_line_detection
import cv2 as cv
import numpy as np
from sklearn.linear_model import LinearRegression
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
    img = cv.imread(filename)
    horizontal_lines = horizontal_line_detection(filename)
    horizontal_lines_list = []
    for line in horizontal_lines:
        x1, y1, x2, y2 = line[0]
        horizontal_lines_list.append((x1, y1))
        horizontal_lines_list.append((x2, y2))

    x = [point[0] for point in horizontal_lines_list]
    y = [point[1] for point in horizontal_lines_list]

    regressor = LinearRegression()
    X = np.array(x).reshape(-1, 1)
    if x and y is not None:
        regressor.fit(X, y)

        slope = regressor.coef_[0]
        intercept = regressor.intercept_

        # Define the starting and ending x-coordinates for the line segment
        start_x = min(x)
        end_x = max(x)

        start_y = int(slope * start_x + intercept)
        end_y = int(slope * end_x + intercept)

        cv.line(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), thickness=2)
        cv.imwrite('cvat_dataset/images/output/' + str(i) + '.png',img)
        i = i +1
    else:
        continue