from detector import horizontal_line_detection, vertical_line_detection
from evaluation import create_content
import numpy as np
import cv2 as cv
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import json
import os
import sys


i = 0
json_content = list()


arg = sys.argv[1:]

v = "-v" in arg

if len(arg) == 0:
    print('No arguments detected. Demo with visualization for 1 image. Output in output.json.')
    image_files = ['SA_20211116-083704_incision_crop_0.jpg']
    output_file = 'output.json'
    v = True
else:
    if v:
        image_files = arg[2:]
    else:
        image_files = arg[1:]
    output_file = arg[0]
v = False

directory = '../cvat_dataset/images/default/'

files = []
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    files.append(f)

for filename in files:
    print(i)
    img = cv.imread(filename)
    img1 = cv.imread(filename)
    h, w = (cv.cvtColor(img, cv.COLOR_BGR2GRAY)).shape
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
        output_horizontal = np.array([[[point1[0], point1[1], point2[0], point2[1]]]], dtype=np.int32)
        output_vertical = np.empty_like(output_horizontal)

        fin_ver = []
        fin = vertical_line_detection(filename)
        if len(fin) is not 0:
            eps = w*0.2  # Maximum distance between two samples to be considered as part of the same neighborhood
            min_samples = 2  # Minimum number of samples required to form a dense region

            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(fin)
            unique_labels = np.unique(labels)

            for label in unique_labels:
                cluster_lines = fin[labels == label]
                average_line = np.mean(cluster_lines, axis=0)
                fin_ver.append(average_line)

        if fin_ver is not None:
            fin_ver = np.array(fin_ver, dtype=np.int32)
            for points in fin_ver:
                x1, y1, x2, y2 = points
                cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                point1 = [x1, y1]
                point2 = [x2, y2]
                output_tmp = np.array([[[point1[0], point1[1], point2[0], point2[1]]]], dtype=np.int32)
                output_vertical = np.append(output_vertical, output_tmp, axis=0)


        image_name = filename.replace('cvat_dataset/images/default/', '')
        information, intersections, intersection_alphas = create_content(image_name, output_horizontal, output_vertical)
        json_content.append(information[0])
        if v:
            plt.imshow(img)
            plt.title("Title: " + filename)
            plt.show()
        i = i + 1
    else:
        continue

with open(output_file, "w", encoding='utf-8') as fw:
    json.dump(json_content, fw, ensure_ascii=False, indent=4)
