import cv2
import numpy as np
from data_manager import Experiment, DataManager, Feature
from tqdm import tqdm
from typing import Sequence, List

def exec_data(exp: Experiment, exp_id: int, idx: int, contour: cv2.Mat):
    exp.result_data['frame_id'][exp_id][idx] = idx
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
    
    tip_pt = exp.config["data"]["tip_pt"]
    exp.result_data['tip_distance'][exp_id][idx] = calc_tip_distance(tip_pt, contour)


def exec_img(exp: Experiment, exp_id: int):
    raw_imgs = exp.experiment_imgs[exp_id]
    base_img = raw_imgs[0]
    
    # --- Check config to decide whether to save images ---
    should_save_images = exp.config.get('img', {}).get('save_processed_images', True)

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
            
            if should_save_images:
                # Draw contour only if we are saving the image
                contour_img = img.copy()
                cv2.drawContours(contour_img, [max_contour], -1, (0, 255, 0), 2)
                exp.result_imgs['contour'][exp_id][idx] = contour_img

        if should_save_images:
            exp.result_imgs['diff'][exp_id][idx] = diff
            exp.result_imgs['gray'][exp_id][idx] = gray
            exp.result_imgs['thresh'][exp_id][idx] = thresh
            # If no contour was found, we still might need to save the original image
            if 'contour' not in exp.result_imgs or exp.result_imgs['contour'][exp_id][idx] is None:
                 exp.result_imgs['contour'][exp_id][idx] = img

        if max_contour is not None:
            exec_data(exp, exp_id, idx, max_contour)

def transform_all(exp: Experiment, exp_id: int):
    frames = len(exp.experiment_imgs[exp_id])
    for frame_id in range(frames):
        transform_scale(exp.config['data'], exp, exp_id, frame_id)

def exec_once(data_manager: DataManager, exp_id: int):
    exec_img(data_manager.experiment, exp_id)
    transform_all(data_manager.experiment, exp_id)

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

def calc_tip_distance(tip_pt, contour):
    points = contour.reshape(-1, 2)
    tip_x, tip_y = tip_pt

    # If the contour is empty, there's no distance to calculate.
    if points.size == 0:
        return 0

    # Find the index of the point on the contour that is vertically closest to the tip.
    top_point_idx = np.argmin(np.abs(points[:, 0] - tip_x))
    top_point = points[top_point_idx]

    # The distance is the difference in the y-coordinates.
    # Since the tip is above the contour, tip_y is smaller, so this should be positive.
    distance = top_point[1] - tip_y
    
    # The distance should not be negative, but as a safeguard.
    return max(0, distance)


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


def calc_vec_direction(curr_dist: List[float], prev_dist: List[float]):
    return [(curr - prev) for curr, prev in zip(curr_dist, prev_dist)]

def transform_scale(data_config, exp: Experiment, exp_id: int, frame_id: int):
    distance_scale = data_config['distance_scale']
    time_scale = data_config['time_scale']

    exp.result_data['area'][exp_id][frame_id] *= distance_scale ** 2 
    exp.result_data['arc_length'][exp_id][frame_id] *= distance_scale
    exp.result_data['area_vec'][exp_id][frame_id] *= distance_scale ** 2 / time_scale
    dists = exp.result_data['expand_dist'][exp_id][frame_id]
    exp.result_data['expand_dist'][exp_id][frame_id] = [d * distance_scale for d in dists]
    vecs = exp.result_data['expand_vec'][exp_id][frame_id]
    exp.result_data['expand_vec'][exp_id][frame_id] = [v * distance_scale / time_scale for v in vecs]
    exp.result_data['time'][exp_id][frame_id] = exp.result_data['frame_id'][exp_id][frame_id] * time_scale
    exp.result_data['tip_distance'][exp_id][frame_id] *= distance_scale