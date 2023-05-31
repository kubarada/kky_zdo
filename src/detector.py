import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



def horizontal_line_detection(PATH_TO_FILE):
    img = cv.imread(PATH_TO_FILE)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h, w = gray.shape
    # resizing, bcs works better on bigger pictures
    if w < 200:
        w = int(w * 2)
        h = int(h * 2)
        gray = cv.resize(gray, (w, h), interpolation=cv.INTER_CUBIC)

    th = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 17, 2)

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


def vertical_line_detection(PATH_TO_FILE):

    img = cv.imread(PATH_TO_FILE)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    gray = cv.GaussianBlur(gray, (17, 17), 0)

    h, w = gray.shape
    w = int(w * 2)
    h = int(h * 2)
    gray = cv.resize(gray, (w, h), interpolation=cv.INTER_CUBIC)

    # Apply adaptive thresholding
    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 17, 2)

    h, w = thresh.shape
    kernel_vertical = cv.getStructuringElement(cv.MORPH_RECT, (1,8))
    eroded = cv.erode(thresh, kernel_vertical, iterations=1)
    thinner_line = cv.dilate(eroded, kernel_vertical, iterations=1)
    kernel_vertical = cv.getStructuringElement(cv.MORPH_RECT, (1, 9))
    thinner_line = cv.dilate(thinner_line, kernel_vertical, iterations=3)
    kernel_vertical = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))
    thinner_line = cv.erode(thinner_line, kernel_vertical, iterations=1)

    # Perform Hough Transform to detect lines
    lines = cv.HoughLinesP(thinner_line, 1, np.pi / 180, 15, h*0.2, h*0.1)
    #print(lines)
    # Identify stitches based on their angle
    stitches = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if np.abs(angle) > 55:
                stitches.append(line)

    # if stitches are empty, try to adjust the angle
    if len(stitches) < 2:
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                if np.abs(angle) > 20:
                    stitches.append(line)
    if img.shape == thresh.shape:
        return stitches
    else:
        ratio = int(thresh.shape[0] / img.shape[0])  # ratio between two images
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

def postprocessing_stitch(keypoints, img):

    if len(keypoints) > 2:
        k_means_in = list()
        for i in range(len(keypoints)):
            points = keypoints[i][0]
            k_means_in.append(points)
        k_means_in = np.array(k_means_in)

        clusters = dict()
        k_values = range(2, len(keypoints))
        silhouette_scores = []

        for k in k_values:
            kmeans = KMeans(n_clusters=k, n_init="auto")
            kmeans.fit(k_means_in)
            labels = kmeans.labels_
            score = silhouette_score(k_means_in, labels)
            silhouette_scores.append(score)

        best_k = k_values[np.argmax(silhouette_scores)]

        kmeans = KMeans(n_clusters=best_k)
        kmeans.fit(k_means_in)
        labels = kmeans.labels_

        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append([k_means_in[i]])

        final_keypoints = list()
        for i in range(0, len(clusters.keys())):
            start_points_inc = list()
            end_points_inc = list()
            for j in range(0, len(clusters[i])):
                current_points = clusters[i][0][0]
                start_part = np.array([current_points[0], current_points[1]])
                end_part = np.array([current_points[2], current_points[3]])
                start_points_inc.append(start_part)
                end_points_inc.append(end_part)
            keypoints_average = average_coordinates(np.array([start_part]), np.array([end_part]))
            final_keypoints.append(keypoints_average)
            start_part = list()
            end_part = list()

        if final_keypoints is not None:
            final_coordinations = list()
            banned_lines = list()
            for index1, line1 in enumerate(final_keypoints):
                for index2 in range(index1 + 1, len(final_keypoints)):
                    line2 = final_keypoints[index2]
                    line1_midpoint = [(line1[0][0][0] + line1[0][0][2]) / 2, (line1[0][0][1] + line1[0][0][3]) / 2]
                    line2_midpoint = [(line2[0][0][0] + line2[0][0][2]) / 2, (line2[0][0][1] + line2[0][0][3]) / 2]
                    distance = calculate_distance(line1_midpoint[0], line1_midpoint[1], line2_midpoint[0],
                                                  line2_midpoint[1])
                    if distance <= img.shape[1] * 0.05:
                        start_points = [[line1[0][0][0], line1[0][0][1]], [line2[0][0][2], line2[0][0][3]]]
                        end_points = [[line1[0][0][2], line1[0][0][3]], [line2[0][0][0], line2[0][0][1]]]
                        one_line = average_coordinates(np.array(start_points), np.array(end_points))
                        final_coordinations.append(one_line)
                        banned_lines.append(line1)
                        banned_lines.append(line2)

            for line1 in final_keypoints:
                if line_in_lines(line1, banned_lines):
                    continue
                else:
                    final_coordinations.append(line1)
            return final_coordinations
        else:
            return None


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
