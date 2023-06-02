import numpy as np
import math

def compute_intersections(incisions, stitches):

    incision_alphas = []
    incision_lines = []

    for incision in incisions:
        for (p_1, p_2) in zip(incision[:-1], incision[1:]):
            p1 = np.array(p_1)
            p2 = np.array(p_2)
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            if dy == 0:
                alpha = 90.0
            elif dx == 0:
                alpha = 0.0
            else:
                alpha = 90 + 180. * np.arctan(dy / dx) / np.pi
            incision_alphas.append(alpha)
            incision_lines.append([p1, p2])

    stitch_alphas = []
    stitch_lines = []

    for stitch in stitches:
        for (p_1, p_2) in zip(stitch[:-1], stitch[1:]):
            p1 = np.array(p_1)
            p2 = np.array(p_2)
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            if dy == 0:
                alpha = 90.0
            elif dx == 0:
                alpha = 180.0
            else:
                alpha = 90 + 180. * np.arctan(dy / dx) / np.pi
            stitch_alphas.append(alpha)
            stitch_lines.append([p1, p2])
    intersections = []
    intersections_alphas = []
    for (incision_line, incision_alpha) in zip(incision_lines, incision_alphas):
        for (stitch_line, stitch_alpha) in zip(stitch_lines, stitch_alphas):

            p0, p1 = incision_line
            pA, pB = stitch_line
            (xi, yi, valid, r, s) = intersectLines(p0, p1, pA, pB)
            if valid == 1:
                intersections.append([format(xi, ".2f"), format(yi, ".2f")])
                alpha_diff = abs(incision_alpha - stitch_alpha)
                alpha_diff = 180.0 - alpha_diff if alpha_diff > 90.0 else alpha_diff
                # alpha_diff = 90 - alpha_diff
                intersections_alphas.append(format(alpha_diff, ".2f"))

    return intersections, intersections_alphas

def intersectLines(pt1, pt2, ptA, ptB):

    DET_TOLERANCE = 0.00000001

    x1, y1 = pt1
    x2, y2 = pt2
    dx1 = x2 - x1
    dy1 = y2 - y1

    x, y = ptA
    xB, yB = ptB
    dx = xB - x
    dy = yB - y

    DET = (-dx1 * dy + dy1 * dx)

    if math.fabs(DET) < DET_TOLERANCE: return (0,0,0,0,0)

    # now, the determinant should be OK
    DETinv = 1.0/DET

    # find the scalar amount along the "self" segment
    r = DETinv * (-dy * (x-x1) + dx * (y-y1))

    # find the scalar amount along the input line
    s = DETinv * (-dy1 * (x-x1) + dx1 * (y-y1))

    # return the average of the two descriptions
    xi = (x1 + r*dx1 + x + s*dx)/2.0
    yi = (y1 + r*dy1 + y + s*dy)/2.0

    valid = 0
    if x1 != x2:
        if x1 < x2:
            a = x1
            b = x2
        else:
            a = x2
            b = x1
        c = xi
    else:
        if y1 < y2:
            a = y1
            b = y2
        else:
            a = y2
            b = y1
        c = yi
    if (c > a) and (c < b):
        if x != xB:
            if x < xB:
                a = x
                b = xB
            else:
                a = xB
                b = x
            c = xi
        else:
            if y < yB:
                a = y
                b = yB
            else:
                a = yB
                b = y
            c = yi
        if (c > a) and (c < b):
            valid = 1

    return xi, yi, valid, r, s

def create_content(image, incision, stitches):

    stitches_in = list()
    incisions_in = list()
    if len(incision) > 0:
        for i in [0,2]:  # start and end
            coordinates = incision[0][0]
            points = [coordinates[i], coordinates[i+1]]
            incisions_in.append(points)
    incisions_in = [incisions_in]
    if stitches is not None:
        for stitch in stitches:
            coordinates = stitch[0]
            line = list()
            for i in [0, 2]:
                points = [coordinates[i], coordinates[i+1]]
                line.append(points)
            stitches_in.append(line)

    intersections, intersections_alphas = compute_intersections(incisions_in, stitches_in)
    information_out = [
        {
                "filename": image,
                "incision_polyline": incision[0][0].tolist(),
                "crossing_positions": intersection_int(intersections),
                "crossing_angles": alphas_int(intersections_alphas)
            },
        ]

    intersections_num = intersection_int(intersections)
    intersections_alphas_num = alphas_int(intersections_alphas)
    return information_out, intersections_num, intersections_alphas_num


def intersection_int(keypoints):
    int_keypoints = list()
    for points in keypoints:
        one_line = list()
        for point in points:
            one_line.append(float(point))
        int_keypoints.append(one_line)
    return int_keypoints

def alphas_int(keypoints):
    int_keypoints = list()
    for point in keypoints:
        int_keypoints.append(float(point))
    return int_keypoints
