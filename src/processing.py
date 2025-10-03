import cv2
from dataloader import Experiment, Dataloader, Feature
from tqdm import tqdm
from typing import Sequence

def exec_data(exp: Experiment, exp_id: int, idx: int, contour: cv2.Mat):
    exp.result_data['contour_pts'][exp_id][idx] = contour
    exp.result_data['area'][exp_id][idx] = cv2.contourArea(contour)
    exp.result_data['arc_length'][exp_id][idx] = cv2.arcLength(contour, True)
    if idx != 0:
        prev_area = exp.result_data['area'][exp_id][idx-1]
        curr_area = exp.result_data['area'][exp_id][idx]
        exp.result_data['expand_vec'][exp_id][idx] = (curr_area - prev_area)
    else:
        exp.result_data['expand_vec'][exp_id][idx] = 0


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
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(img, [max_contour], -1, (0, 255, 0), 2)

        exp.result_imgs['diff'][exp_id][idx] = diff
        exp.result_imgs['gray'][exp_id][idx] = gray
        exp.result_imgs['thresh'][exp_id][idx] = thresh
        exp.result_imgs['contour'][exp_id][idx] = img

        exec_data(exp, exp_id, idx, max_contour)


def exec_once(dataloader: Dataloader, exp_id: int):
    exec_img(dataloader.experiment, 0)