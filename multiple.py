import os
from detector import horizontal_line_detection, postprocessing_stitch, vertical_line_detection
from evaluation import create_content
import numpy as np
import cv2 as cv
from sklearn.linear_model import LinearRegression
import json

directory = 'cvat_dataset/images/default/'

i = 0
json_content = list()

files = []
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    files.append(f)

for filename in files:
    img = cv.imread(filename)
    img1 = cv.imread(filename)
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

        point1 = [start_x, start_y]
        point2 = [end_x, end_y]

        # Append more lines if needed
        output = np.array([[[point1[0], point1[1], point2[0], point2[1]]]], dtype=np.int32)

        fin = vertical_line_detection(filename)
        if fin is not None:
            fin = postprocessing_stitch(fin, img1)
            if fin is not None:
                print('Fin is: ', fin)
                for points in fin:
                    # Extracted points nested in the list
                    test = points[0][0]
                    print('test ', test)
                    x1, y1, x2, y2 = points[0][0]

                    # Draw the lines joing the points
                    # On the original image
                    cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        image_name = filename.replace('cvat_dataset/images/default/', '')
        information, intersections, intersection_alphas = create_content(image_name, output, fin)
        json_content.append(information[0])
        cv.imwrite('cvat_dataset/images/output/' + str(i) + '.png', img)

        i = i + 1
    else:
        continue
print(json_content)
with open('cvat_dataset/output.json', "w", encoding='utf-8') as fw:
    json.dump(json_content, fw, ensure_ascii=False, indent=4)

