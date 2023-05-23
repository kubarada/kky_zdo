from detector import horizontal_line_detection, keypoints_postprocessing
import cv2 as cv
import numpy as np
from sklearn.linear_model import LinearRegression

PATH_TO_FILE = 'cvat_dataset/images/default/SA_20211013-055124_incision_crop_0.jpg'

img = cv.imread(PATH_TO_FILE)
horizontal_lines = horizontal_line_detection(PATH_TO_FILE)
horizontal_lines_list = []
for line in horizontal_lines:
    x1, y1, x2, y2 = line[0]
    horizontal_lines_list.append((x1,y1))
    horizontal_lines_list.append((x2, y2))

x = [point[0] for point in horizontal_lines_list]
y = [point[1] for point in horizontal_lines_list]

regressor = LinearRegression()
X = np.array(x).reshape(-1, 1)
regressor.fit(X, y)

slope = regressor.coef_[0]
intercept = regressor.intercept_

# Define the starting and ending x-coordinates for the line segment
start_x = min(x)
end_x = max(x)

start_y = int(slope * start_x + intercept)
end_y = int(slope * end_x + intercept)

cv.line(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), thickness=2)
cv.imwrite('detectedLinesHorizontal.png', img)
