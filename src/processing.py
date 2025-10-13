import cv2
import numpy as np
from data_manager import Experiment, DataManager, Feature
from tqdm import tqdm
from typing import Sequence

def exec_data(exp: Experiment, exp_id: int, idx: int, contour: cv2.Mat):
    exp.result_data['contour_pts'][exp_id][idx] = contour
    exp.result_data['area'][exp_id][idx] = cv2.contourArea(contour)
    exp.result_data['arc_length'][exp_id][idx] = cv2.arcLength(contour, True)
    if idx != 0:
        prev_area = exp.result_data['area'][exp_id][idx-1]
        curr_area = exp.result_data['area'][exp_id][idx]
        exp.result_data['area_vec'][exp_id][idx] = (curr_area - prev_area)
    else:
        exp.result_data['area_vec'][exp_id][idx] = 0
    # regression circle
    center, radius = regression_circle(contour)
    exp.result_data['regression_circle_center'][exp_id][idx] = center
    exp.result_data['regression_circle_radius'][exp_id][idx] = radius
    # expand velocity every direction
    hole_pt = exp.config["data"]["hole_pt"]
    exp.result_data['expand_dist'][exp_id][idx] = calc_dist_direction(hole_pt, contour)
    if idx != 0:
        prev_dist = exp.result_data['expand_dist'][exp_id][idx-1]
        curr_dist = exp.result_data['expand_dist'][exp_id][idx]
        exp.result_data['expand_vec'][exp_id][idx] = calc_vec_direction(curr_dist, prev_dist)
    else:
        exp.result_data['expand_vec'][exp_id][idx] = [0.,0.,0.,0.]


def exec_img(exp: Experiment, exp_id: int):
    raw_imgs = exp.experiment_imgs[exp_id]
    base_img = raw_imgs[0]
    progress_bar = tqdm(enumerate(raw_imgs), 
                        total=len(raw_imgs), 
                        desc=f"process exp id = {exp_id}",
                        unit="img")
    for idx, img in progress_bar:
        diff = cv2.absdiff(base_img, img)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_contour = None
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(img, [max_contour], -1, (0, 255, 0), 2)

        exp.result_imgs['diff'][exp_id][idx] = diff
        exp.result_imgs['gray'][exp_id][idx] = gray
        exp.result_imgs['thresh'][exp_id][idx] = thresh
        exp.result_imgs['contour'][exp_id][idx] = img

        if max_contour is not None:
            exec_data(exp, exp_id, idx, max_contour)


def exec_once(data_manager: DataManager, exp_id: int):
    exec_img(data_manager.experiment, 0)

def regression_circle(contour: cv2.Mat):
    points = contour.reshape(-1, 2)
    x = points[:, 0]
    y = points[:, 1]

    A = np.vstack([x * 2, y * 2, np.ones(len(x))]).T
    b = x**2 + y**2

    solution, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, b, c = solution

    radius = np.sqrt(c + a**2 + b**2)

    center = (int(a), int(b))
    radius = int(radius)

    return center, radius

def calc_dist_direction(hole_pt, contour):
    points = contour.reshape(-1, 2)
    hole_x, hole_y = hole_pt

    # Points in each direction
    points_above = points[points[:, 1] < hole_y]
    points_below = points[points[:, 1] > hole_y]
    points_left = points[points[:, 0] < hole_x]
    points_right = points[points[:, 0] > hole_x]

    dist_up, dist_down, dist_left, dist_right = 0, 0, 0, 0

    # For 'up', we want the point in points_above that is closest to the vertical line x=hole_x
    if points_above.size > 0:
        top_point_idx = np.argmin(np.abs(points_above[:, 0] - hole_x))
        top_point = points_above[top_point_idx]
        dist_up = hole_y - top_point[1]

    # For 'down', we want the point in points_below that is closest to the vertical line x=hole_x
    if points_below.size > 0:
        bottom_point_idx = np.argmin(np.abs(points_below[:, 0] - hole_x))
        bottom_point = points_below[bottom_point_idx]
        dist_down = bottom_point[1] - hole_y

    # For 'left', we want the point in points_left that is closest to the horizontal line y=hole_y
    if points_left.size > 0:
        left_point_idx = np.argmin(np.abs(points_left[:, 1] - hole_y))
        left_point = points_left[left_point_idx]
        dist_left = hole_x - left_point[0]

    # For 'right', we want the point in points_right that is closest to the horizontal line y=hole_y
    if points_right.size > 0:
        right_point_idx = np.argmin(np.abs(points_right[:, 1] - hole_y))
        right_point = points_right[right_point_idx]
        dist_right = right_point[0] - hole_x

    return [dist_up, dist_right, dist_down, dist_left]


def calc_vec_direction(curr_dist, prev_dist):
    return [(curr - prev) for curr, prev in zip(curr_dist, prev_dist)]