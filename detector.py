import numpy as np
import skimage.feature
import matplotlib.pyplot as plt
import cv2 as cv
import math
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_score


def horizontal_line_detection(PATH_TO_FILE):
    img = cv.imread(PATH_TO_FILE)  # TODO zkombinovat do jednoho radku
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h, w = gray.shape
    # resizing, bcs works better on bigger pictures
    if w < 200:
        w = int(w * 2)
        h = int(h * 2)
        gray = cv.resize(gray, (w, h), interpolation=cv.INTER_CUBIC)

    th = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 17, 2)
    # TODO add morphological operations

    lines = cv.HoughLinesP(th, 1, np.pi / 180, 150, w * 0.75, w*0.1)
    lines_list = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi) < 30:  # adding lines with just certain angle
                lines_list.append(line)
    else:
        # SECOND TRY - resizing, bcs works better on bigger pictures
        w = int(w * 2)
        h = int(h * 2)
        gray = cv.resize(gray, (w, h), interpolation=cv.INTER_CUBIC)

        th = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 17, 2)
        lines = cv.HoughLinesP(th, 1, np.pi / 180, 150, w * 0.75, w * 0.1)

        if lines is not None:

            for line in lines:
                x1, y1, x2, y2 = line[0]
                if np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi) < 30:  # adding lines with just certain angle
                    lines_list.append(line)
        else:
            print('No lines to detect in image ' + str(PATH_TO_FILE))

    if img.shape == gray.shape:
        return lines_list
    else:
        ratio = int(gray.shape[0] / img.shape[0])  # ratio between two images
        for i in range(0, len(lines_list)):
            lines_list[i][0] = (lines_list[i][0] / ratio).astype(np.int32)
        return lines_list

def detect_stitches(image, false_detected_stitches):

    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    kernel_size = 17
    gray = cv.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    scale_percent = 200
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    dim = (width, height)
    gray = cv.resize(gray, dim, interpolation=cv.INTER_CUBIC)
    out = cv.resize(img, dim, interpolation=cv.INTER_CUBIC)

    # Apply adaptive thresholding
    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 17, 2)

    # Determine the mean value of the binary image
    mean_value = np.mean(thresh)

    # Set Canny parameters based on the mean value
    low_threshold = int(mean_value * 0.5)
    high_threshold = int(mean_value * 1)

    # Apply Canny edge detection
    #edges = cv.Canny(thresh, low_threshold, high_threshold)
    dims = thresh.shape
    kernel_vertical = cv.getStructuringElement(cv.MORPH_RECT, (1, 5))
    edges = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel_vertical)
    edges = cv.dilate(edges, kernel_vertical, iterations=8)
    kernel_vertical = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))
    edges = cv.erode(edges, kernel_vertical, iterations=3)
    cv.imwrite('canny.png', edges)

    # Perform Hough Transform to detect lines
    lines = cv.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=None, minLineLength=dims[0]*0.3, maxLineGap=dims[0]*0.2)

    # Identify stitches based on their angle
    stitches = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if np.abs(angle) > 20:
                stitches.append(line)
    else:
        false_detected_stitches += 1

    # if stitches are empty, try to adjust the angle
    if len(stitches) < 2:
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                if np.abs(angle) > 20:
                    stitches.append(line)
        else:
            false_detected_stitches += 1
    if img.shape == gray.shape:
        return stitches
    else:
        ratio = int(gray.shape[0] / img.shape[0])  # ratio between two images
        for i in range(0, len(stitches)):
            stitches[i][0] = (stitches[i][0] / ratio).astype(np.int32)
        return stitches

def average_coordinates(start_points, end_points):
    keypoints_out = list()
    # controlling if any incisions and stitches were detected
    if len(start_points) != 0 and len(end_points) != 0:
        start_x = np.mean(start_points[:, 0]).astype(np.int32)
        start_y = np.mean(start_points[:, 1]).astype(np.int32)
        end_x = np.mean(end_points[:, 0]).astype(np.int32)
        end_y = np.mean(end_points[:, 1]).astype(np.int32)

        # returning the proper format
        keypoints_out.append(np.array([[start_x, start_y, end_x, end_y]]))

    return keypoints_out

def keypoints_postprocessing(keypoints, img, keypoints_type, image):
    print(image)
    if keypoints_type == "incision":
        threshold_band = img.shape[0]*0.05
        start_points = list()  # for start points
        end_points = list()  # for end points

        # for the final output
        start_points_out = list()
        end_points_out = list()
        far_keypoints = list()
        if len(keypoints) > 1:
            for line_part in [0, 2]:  # starts then ends
                for i in range(0, len(keypoints)):  # getting start/end points (x,y)
                    current_points = keypoints[i][0]  # x1,y1,x2,y2
                    curr_part = np.array([current_points[line_part], current_points[line_part+1]])

                    # store the corresponding coordinates
                    if line_part == 0:
                        start_points.append(curr_part)
                    else:
                        end_points.append(curr_part)

            # making the average of the detected coordinates (x,y)
            keypoints_out = average_coordinates(np.array(start_points), np.array(end_points))

            # computing the reference points
            x_start = keypoints_out[0][0][0]
            y_start = keypoints_out[0][0][1]
            x_end = keypoints_out[0][0][2]
            y_end = keypoints_out[0][0][3]


            for line_part in [0, 2]:  # starts then ends
                for i in range(0, len(keypoints)):  # getting start/end points (x,y)
                    current_points = keypoints[i][0]  # x1,y1,x2,y2
                    curr_part = np.array([current_points[line_part], current_points[line_part+1]])  # x1,y1 or x2,y2
                    y_current = curr_part[1]  # y real

                    if line_part == 0:  # detecting the start points
                        if np.abs(y_start - y_current) <= threshold_band:
                            start_points_out.append(curr_part)
                        else:
                            start_points_out.append(curr_part)
                    else:
                        if np.abs(y_end - y_current) <= threshold_band:
                            end_points_out.append(curr_part)
                        else:
                            end_points_out.append(curr_part)

            # compute the final coordinates
            keypoints_out = average_coordinates(np.array(start_points_out), np.array(end_points_out))
            if len(keypoints_out) == 0:
                print(image)
            return keypoints_out
        else:
            return keypoints  # returning the (x1,y1) (x2,y2)

    elif keypoints_type == "stitch" and len(keypoints) > 2:
        k_means_in = list()
        for i in range(len(keypoints)):
            points = keypoints[i][0]
            k_means_in.append(points)
        k_means_in = np.array(k_means_in)

        clusters = dict()
        # identify the classes with corresponding coordinated
        k_values = range(2, len(keypoints))  # Range of k values to try
        silhouette_scores = []  # List to store silhouette scores

        # Perform K-means clustering for different values of k
        for k in k_values:
            kmeans = KMeans(n_clusters=k, n_init="auto")
            kmeans.fit(k_means_in)
            labels = kmeans.labels_
            score = silhouette_score(k_means_in, labels)
            silhouette_scores.append(score)

        # Find the optimal number of clusters
        best_k = k_values[np.argmax(silhouette_scores)]

        # Perform K-means clustering with the best k
        kmeans = KMeans(n_clusters=best_k)
        kmeans.fit(k_means_in)
        labels = kmeans.labels_

        # Get the classes and corresponding values
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append([k_means_in[i]])

        # preparing and averaging the detected classes of incisions
        final_keypoints = list()
        for i in range(0, len(clusters.keys())):
            start_points_inc = list()
            end_points_inc = list()
            for j in range(0, len(clusters[i])):
                current_points = clusters[i][0][0]  # x1,y1,x2,y2
                start_part = np.array([current_points[0], current_points[1]])  # x1,y1
                end_part = np.array([current_points[2], current_points[3]])  # x2,y2
                start_points_inc.append(start_part)
                end_points_inc.append(end_part)
            keypoints_average = average_coordinates(np.array([start_part]), np.array([end_part]))
            final_keypoints.append(keypoints_average)
            start_part = list()
            end_part = list()

        return final_keypoints
def coordinates_control(keypoints, img, image):
    threshold = img.shape[1]*0.05
    keypoints_final = list()
    banned_lines = list()
    for index1, line1 in enumerate(keypoints):
        for index2 in range(index1+1, len(keypoints)):
            line2 = keypoints[index2]
            line1_midpoint = [(line1[0][0][0] + line1[0][0][2]) / 2, (line1[0][0][1] + line1[0][0][3]) / 2]
            line2_midpoint = [(line2[0][0][0] + line2[0][0][2]) / 2, (line2[0][0][1] + line2[0][0][3]) / 2]
            distance = calculate_distance(line1_midpoint[0], line1_midpoint[1], line2_midpoint[0], line2_midpoint[1])
            if distance <= threshold:
                start_points = [[line1[0][0][0], line1[0][0][1]], [line2[0][0][2], line2[0][0][3]]]
                end_points = [[line1[0][0][2], line1[0][0][3]], [line2[0][0][0], line2[0][0][1]]]
                one_line = average_coordinates(np.array(start_points), np.array(end_points))
                keypoints_final.append(one_line)
                banned_lines.append(line1)
                banned_lines.append(line2)

    for line1 in keypoints:
        if line_in_lines(line1, banned_lines):
            continue
        else:
            keypoints_final.append(line1)
    return keypoints_final


def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def line_in_lines(line, keypoints):
    for points in keypoints:
        if len(points) > 1:
            for point in points:
                if point == line:
                    return True
                    break
        elif np.array_equal(points[0][0], line[0][0]):
            return True
            break
    return False
