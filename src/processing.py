import cv2
import numpy as np
from data_manager import Experiment, DataManager, Feature
from tqdm import tqdm
from typing import Sequence, List

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
    
    tip_pt = exp.config["data"]["tip_pt"]
    exp.result_data['tip_distance'][exp_id][idx] = calc_tip_distance(tip_pt, contour)


def exec_img(exp: Experiment, exp_id: int):
    raw_imgs = exp.experiment_imgs[exp_id]
    base_img = raw_imgs[0]
    
    img_config = exp.config.get('img', {})
    should_save_images = img_config.get('save_processed_images', True)
    min_contour_area = img_config.get('min_contour_area', 0)
    normalize_gray = img_config.get('normalize_gray', False)
    normalize_diff_output = img_config.get('normalize_diff_output', False)
    threshold_method = img_config.get('threshold_method', 'fixed').lower()
    binary_threshold = img_config.get('binary_threshold', 127)
    adaptive_block_size = img_config.get('adaptive_block_size', 51)
    adaptive_C = img_config.get('adaptive_C', 2)
    blur_kernel_size = img_config.get('blur_kernel_size', 0)
    apply_morph_close = img_config.get('apply_morph_close', False)
    morph_kernel_size = img_config.get('morph_kernel_size', 3)
    morph_iterations = img_config.get('morph_iterations', 1)

    # === 新增：顶端闭合与洪泛闭合的配置（可不配，有默认值） ===
    top_close_enable = img_config.get('top_close_enable', True)
    top_close_height_ratio = img_config.get('top_close_height_ratio', 0.25)
    top_close_kernel_w = img_config.get('top_close_kernel_w', 3)
    top_close_kernel_h = img_config.get('top_close_kernel_h', 11)
    use_floodfill_close = img_config.get('use_floodfill_close', False)

    if isinstance(blur_kernel_size, (int, float)):
        blur_kernel_size = int(blur_kernel_size)
        if blur_kernel_size < 3:
            blur_kernel_size = 0
        elif blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
    else:
        blur_kernel_size = 0

    if isinstance(morph_kernel_size, (int, float)):
        morph_kernel_size = max(1, int(morph_kernel_size))
        if morph_kernel_size % 2 == 0:
            morph_kernel_size += 1
    else:
        morph_kernel_size = 1

    morph_iterations = max(1, int(morph_iterations)) if isinstance(morph_iterations, (int, float)) else 1

    progress_bar = tqdm(enumerate(raw_imgs), 
                        total=len(raw_imgs), 
                        desc=f"process exp id = {exp_id}",
                        unit="img")
    for idx, img in progress_bar:
        exp.result_data['frame_id'][exp_id][idx] = idx
        diff = cv2.absdiff(base_img, img)
        diff_to_save = diff
        if normalize_diff_output:
            diff_to_save = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        processed_gray = gray
        if normalize_gray:
            processed_gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        gray_for_threshold = processed_gray
        if blur_kernel_size:
            gray_for_threshold = cv2.GaussianBlur(gray_for_threshold, (blur_kernel_size, blur_kernel_size), 0)

        if threshold_method == 'otsu':
            ret, thresh = cv2.threshold(gray_for_threshold, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        elif threshold_method == 'adaptive':
            block_size = max(3, adaptive_block_size)
            if block_size % 2 == 0:
                block_size += 1
            thresh = cv2.adaptiveThreshold(
                gray_for_threshold,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                adaptive_C
            )
        else:
            ret, thresh = cv2.threshold(gray_for_threshold, binary_threshold, 255, cv2.THRESH_BINARY)

        if apply_morph_close and morph_kernel_size > 1:
            kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)

        # ==== 顶部 ROI 竖向闭合（弥合顶端开口） ====
        if top_close_enable:
            H, W = thresh.shape
            y2 = int(max(1, min(H, int(H * float(top_close_height_ratio)))))
            roi = thresh[:y2, :].copy()
            vker = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (int(max(1, top_close_kernel_w)), int(max(1, top_close_kernel_h)))
            )
            roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, vker, iterations=max(1, int(morph_iterations)))
            roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
            thresh[:y2, :] = roi

        # ==== 兜底：外部洪泛填充闭合（可选） ====
        if use_floodfill_close:
            inv = (255 - thresh).astype(np.uint8)
            h, w = inv.shape
            mask = np.zeros((h+2, w+2), np.uint8)
            cv2.floodFill(inv, mask, (0, 0), 0)
            filled = 255 - inv
            thresh = filled
        
        top_convexify = True
        top_convex_height_ratio = img_config.get('top_convex_height_ratio', 0.28)  # 顶部条带比例
        if top_convexify:
            H, W = thresh.shape
            y2 = int(max(1, min(H, int(H * float(top_convex_height_ratio)))))
            top = thresh[:y2, :].copy()

            # 仅在顶部条带收集前景点
            cnts, _ = cv2.findContours(top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                pts = np.vstack([c.reshape(-1, 2) for c in cnts])   # (x,y)
                if len(pts) >= 3:
                    hull = cv2.convexHull(pts)                      # 顶部点的凸包
                    top_filled = np.zeros_like(top)
                    cv2.fillConvexPoly(top_filled, hull, 255)

                    # 可选：限制在原有火焰包围盒内，避免“长出去”
                    x, y, w, h = cv2.boundingRect(np.vstack(cnts))
                    mask_bbox = np.zeros_like(top)
                    mask_bbox[y:y+h, x:x+w] = 255
                    top_filled = cv2.bitwise_and(top_filled, mask_bbox)

                    # 轻微平滑边缘，避免锯齿
                    top_filled = cv2.morphologyEx(
                        top_filled, cv2.MORPH_CLOSE,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations=1
                    )

                    # 用凸包结果替换顶部条带
                    thresh[:y2, :] = top_filled

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_contour = None
        if contours:
            candidate_contour = max(contours, key=cv2.contourArea)
            candidate_area = cv2.contourArea(candidate_contour)

            if should_save_images:
                contour_img = img.copy()
                cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1)
                if candidate_area >= min_contour_area:
                    cv2.drawContours(contour_img, [candidate_contour], -1, (0, 0, 255), 2)
                exp.result_imgs['contour'][exp_id][idx] = contour_img

            if candidate_area >= min_contour_area:
                max_contour = candidate_contour

        if should_save_images:
            exp.result_imgs['raw'][exp_id][idx] = img
            exp.result_imgs['diff'][exp_id][idx] = diff_to_save
            exp.result_imgs['gray'][exp_id][idx] = processed_gray
            exp.result_imgs['thresh'][exp_id][idx] = thresh
            # 如果未画出轮廓，确保有占位图
            if exp.result_imgs.get('contour') is not None:
                if exp.result_imgs['contour'][exp_id][idx] is None:
                    exp.result_imgs['contour'][exp_id][idx] = np.zeros_like(img)

        if max_contour is not None:
            exec_data(exp, exp_id, idx, max_contour)

def transform_all(exp: Experiment, exp_id: int):
    frames = len(exp.experiment_imgs[exp_id])
    for frame_id in range(frames):
        transform_scale(exp.config['data'], exp, exp_id, frame_id)

def exec_once(data_manager: DataManager, exp_id: int):
    exec_img(data_manager.experiment, exp_id)
    transform_all(data_manager.experiment, exp_id)
    trim_experiment_to_area_peak(data_manager.experiment, exp_id)

def regression_circle(contour: cv2.Mat):
    points = contour.reshape(-1, 2)
    x = points[:, 0]
    y = points[:, 1]

    A = np.vstack([x * 2, y * 2, np.ones(len(x))]).T
    b = x**2 + y**2

    solution, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, b_, c = solution

    radius = np.sqrt(c + a**2 + b_**2)

    center = (int(a), int(b_))
    radius = int(radius)

    return center, radius

def calc_tip_distance(tip_pt, contour):
    points = contour.reshape(-1, 2)
    tip_x, tip_y = tip_pt

    if points.size == 0:
        return 0

    # 找到与 tip_x 横向最接近的轮廓点
    top_point_idx = np.argmin(np.abs(points[:, 0] - tip_x))
    top_point = points[top_point_idx]

    distance = top_point[1] - tip_y
    return max(0, distance)

def calc_dist_direction(hole_pt, contour):
    points = contour.reshape(-1, 2)
    hole_x, hole_y = hole_pt

    points_above = points[points[:, 1] < hole_y]
    points_below = points[points[:, 1] > hole_y]
    points_left  = points[points[:, 0] < hole_x]
    points_right = points[points[:, 0] > hole_x]

    dist_up, dist_down, dist_left, dist_right = 0, 0, 0, 0

    if points_above.size > 0:
        top_point_idx = np.argmin(np.abs(points_above[:, 0] - hole_x))
        top_point = points_above[top_point_idx]
        dist_up = hole_y - top_point[1]

    if points_below.size > 0:
        bottom_point_idx = np.argmin(np.abs(points_below[:, 0] - hole_x))
        bottom_point = points_below[bottom_point_idx]
        dist_down = bottom_point[1] - hole_y

    if points_left.size > 0:
        left_point_idx = np.argmin(np.abs(points_left[:, 1] - hole_y))
        left_point = points_left[left_point_idx]
        dist_left = hole_x - left_point[0]

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

def trim_experiment_to_area_peak(exp: Experiment, exp_id: int):
    area_series = exp.result_data['area'][exp_id]
    if not area_series:
        return

    area_array = np.array(area_series)
    non_zero_indices = np.where(area_array > 0)[0]
    if non_zero_indices.size == 0:
        return

    start_idx = int(non_zero_indices[0])
    peak_value = area_array.max()
    peak_indices = np.where(area_array == peak_value)[0]
    if peak_indices.size == 0:
        return
    end_idx = int(peak_indices[-1])

    if end_idx < start_idx:
        end_idx = start_idx

    slice_obj = slice(start_idx, end_idx + 1)

    exp.experiment_imgs[exp_id] = exp.experiment_imgs[exp_id][slice_obj]
    if exp.experiment_features[exp_id]:
        exp.experiment_features[exp_id] = exp.experiment_features[exp_id][slice_obj]

    for result_type, per_experiment_imgs in exp.result_imgs.items():
        exp.result_imgs[result_type][exp_id] = per_experiment_imgs[exp_id][slice_obj]

    for key, per_experiment_data in exp.result_data.items():
        exp.result_data[key][exp_id] = per_experiment_data[exp_id][slice_obj]

    frame_ids = exp.result_data['frame_id'][exp_id]
    if frame_ids:
        base_frame = frame_ids[0]
        exp.result_data['frame_id'][exp_id] = [fid - base_frame for fid in frame_ids]

    times = exp.result_data['time'][exp_id]
    if times:
        base_time = times[0]
        exp.result_data['time'][exp_id] = [t - base_time for t in times]

    area_vec = exp.result_data['area_vec'][exp_id]
    if area_vec:
        area_vec[0] = 0.0

    expand_vec = exp.result_data['expand_vec'][exp_id]
    if expand_vec:
        expand_vec[0] = [0.0, 0.0, 0.0, 0.0]
