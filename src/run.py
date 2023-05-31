import os
from detector import horizontal_line_detection, postprocessing_stitch, vertical_line_detection
from evaluation import create_content
import numpy as np
import cv2 as cv
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import json
import sys


i = 0
json_content = list()


arg = sys.argv[1:]

v = "-v" in arg

if len(arg) == 0:
    print('No arguments detected. Demo with visualization for 1 image. Output in output.json.')
    image_files = ['../cvat_dataset/images/default/SA_20220620-103348_incision_crop_0.jpg']
    output_file = 'output.json'
    v = True
else:
    if v:
        image_files = arg[2:]
    else:
        image_files = arg[1:]
    output_file = arg[0]

for filename in image_files:
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
                #print('Fin is: ', fin)
                for points in fin:
                    # Extracted points nested in the list
                    test = points[0][0]
                    #print('test ', test)
                    x1, y1, x2, y2 = points[0][0]

                    # Draw the lines joing the points
                    # On the original image
                    cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        image_name = filename.replace('cvat_dataset/images/default/', '')
        information, intersections, intersection_alphas = create_content(image_name, output, fin)
        json_content.append(information[0])
        if v:
            plt.imshow(img)
            plt.title("Title: " + filename)
            plt.show()


        i = i + 1
    else:
        continue
#print(json_content)
with open(output_file, "w", encoding='utf-8') as fw:
    json.dump(json_content, fw, ensure_ascii=False, indent=4)
