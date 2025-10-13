import cv2
import os
from typing import Dict, List
from dataclasses import dataclass, asdict
from loguru import logger
import ipdb
from tqdm import tqdm
import json
import numpy as np

from config import Config

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class Feature:
    def __init__(self, feature_dict: dict, label_key: str = None):
        self.feature_dict: dict = feature_dict
        self.label_key: str = label_key
        if label_key is not None:
            self.label = feature_dict[label_key]
            self.features = {k: v for k, v in feature_dict.items() if k != label_key}
        else:
            self.label = None
            self.features = feature_dict

@dataclass
class ExperimentCondition:
    gas_type: str
    gas_percent: float
    pressure1: float
    pressure2: float
    duration: float

class Experiment:
    def __init__(self, config):
        self.config = config
        self.experiment_imgs: List[List[cv2.Mat]] = []
        self.experiment_features: List[List[Feature]] = []
        self.experiment_conditions: List[ExperimentCondition] = []
        self.experiment_cnt: int = 0
        self.curr_experiment_idx: int = -1
        self.result_imgs: dict = {}
        self.result_data: dict = {}
    
    def add_experiment(self, condition: ExperimentCondition):
        self.experiment_imgs.append([])
        self.experiment_features.append([])
        self.experiment_conditions.append(condition)
        self.curr_experiment_idx += 1
    
    def add_experiment_img(self, img: cv2.Mat):
        self.experiment_imgs[self.curr_experiment_idx].append(img)
    
    def init_result_imgs(self):
        copy_shape = lambda : [[None for _ in experiment_group] for experiment_group in self.experiment_imgs]
        self.result_imgs = {
            'diff': copy_shape(),
            'gray': copy_shape(),
            'thresh': copy_shape(),
            'contour': copy_shape(),
        }
        copy_shape_zero = lambda : [[0.0 for _ in experiment_group] for experiment_group in self.experiment_imgs]
        copy_shape_arr = lambda : [[[] for _ in experiment_group] for experiment_group in self.experiment_imgs]
        self.result_data = {
            'contour_pts': copy_shape_arr(),
            'area': copy_shape_zero(),
            'arc_length': copy_shape_zero(),
            'area_vec': copy_shape_zero(),
            'regression_circle_center': copy_shape_arr(),
            'regression_circle_radius': copy_shape_zero(),
            'expand_dist': copy_shape_arr(),
            'expand_vec': copy_shape_arr(),
            'frame_id': copy_shape_zero(),
            'time': copy_shape_zero(),
        }
    
    def save_result_imgs(self, base_output_dir: str):
        logger.info(f"saving result imgs to '{base_output_dir}'...")
        total_images_to_save = sum(
            1
            for all_exps in self.result_imgs.values()
            for exp_group in all_exps
            for img in exp_group
            if img is not None
        )
        os.makedirs(base_output_dir, exist_ok=True)

        with tqdm(total=total_images_to_save, desc="saving imgs", unit="img") as pbar:
            for result_type, all_experiments_imgs in self.result_imgs.items():
                for exp_id, experiment_group_imgs in enumerate(all_experiments_imgs):
                    for img_idx, img in enumerate(experiment_group_imgs):
                        if img is not None:
                            target_dir = os.path.join(base_output_dir, f"experiment_{exp_id}", result_type)
                            os.makedirs(target_dir, exist_ok=True)
                            
                            file_path = os.path.join(target_dir, f"{img_idx:03d}.png")
                            cv2.imwrite(file_path, img)
                            
                            pbar.update(1)
        
        logger.success("finished")

    def save_result_data(self, base_output_dir: str):
        logger.info(f"saving result data to '{base_output_dir}'...")
        os.makedirs(base_output_dir, exist_ok=True)

        total_frames = sum(len(exp_frames) for exp_frames in self.experiment_imgs)
        
        with tqdm(total=total_frames, desc="saving data", unit="frame") as pbar:
            for exp_id, exp_frames in enumerate(self.experiment_imgs):
                condition = self.experiment_conditions[exp_id]
                condition_data = asdict(condition)
                
                target_dir = os.path.join(base_output_dir, f"experiment_{exp_id}")
                os.makedirs(target_dir, exist_ok=True)

                for frame_id in range(len(exp_frames)):
                    data = condition_data.copy()
                    # data["frame_id"] = frame_id
                    for key in self.result_data:
                        data[key] = self.result_data[key][exp_id][frame_id]

                    file_path = os.path.join(target_dir, f"{frame_id:03d}.json")
                    with open(file_path, 'w') as f:
                        json.dump(data, f, indent=4, cls=NpEncoder)
                    pbar.update(1)
        
        logger.success("finished saving data")

        

class DataManager:
    def __init__(self, config: Config):
        self.config: dict = config.get_config()
        self.img_input_folder: str = self.config['img']['img_input_dir']
        self.img_output_folder: str = self.config['img']['img_output_dir']
        self.result_output_folder: str = self.config['img']['result_output_dir']

        self.experiment = Experiment(self.config)
    
    def parse_condition(self, condition_str: str) -> ExperimentCondition:
        base_name = os.path.basename(condition_str)
        parts = base_name.split('-')
        return ExperimentCondition(
            gas_type=parts[0],
            gas_percent=float(parts[1]),
            pressure1=float(parts[2].replace('Mpa', '')),
            pressure2=float(parts[3].replace('Mpa', '')),
            duration=float(parts[4].replace('ms', ''))
        )
    
    def load_all_experiment(self):
        folders = os.listdir(self.img_input_folder)
        logger.debug(f"{len(folders)} folders found.")
        for folder in folders:
            logger.debug(f"Begin loading experiment {folder}")

            experiment_path = os.path.join(self.img_input_folder, folder)
            files = os.listdir(experiment_path)
            files = [f for f in files if f.endswith('.jpg')]
            logger.debug(f"{len(files)} imgs found.") 

            condition = self.parse_condition(folder)
            logger.debug(f"condition = {condition}")
            self.experiment.add_experiment(condition)

            for file in tqdm(files, desc=f"loading imgs from {folder}"):
                img = cv2.imread(os.path.join(experiment_path, file))
                self.experiment.add_experiment_img(img)
            
            # FIXME: this `break` is only for test
            break
        self.experiment.init_result_imgs()
    
    def save_all_experiment(self):
        # FIXME: this commet is only for test
        # self.experiment.save_result_imgs(self.img_output_folder)
        self.experiment.save_result_data(self.result_output_folder)


