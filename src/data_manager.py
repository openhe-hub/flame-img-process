import cv2
import os
from typing import Dict, List
from dataclasses import dataclass, asdict
from loguru import logger
import ipdb
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter # <-- 1. IMPORT: Added Savitzky-Golay filter

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
        self.experiment_names: List[str] = []
        self.experiment_cnt: int = 0
        self.curr_experiment_idx: int = -1
        self.result_imgs: dict = {}
        self.result_data: dict = {}
    
    def add_experiment(self, condition: ExperimentCondition, name: str):
        self.experiment_imgs.append([])
        self.experiment_features.append([])
        self.experiment_conditions.append(condition)
        self.experiment_names.append(name)
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
            'tip_distance': copy_shape_zero(),
        }
    
    # --- NEW METHOD FOR FILTERING ---
    # VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
    def apply_savitzky_golay_filter(self, filter_configs: dict):
        """
        Applies a Savitzky-Golay filter to smooth the result data using custom settings for each data key.
        
        Args:
            filter_configs (dict): A dictionary where keys are the names of the data series to filter,
                                 and values are dicts with 'window_length' and 'polyorder'.
        """
        logger.info(f"Applying custom Savitzky-Golay filters...")
        
        keys_to_filter = list(filter_configs.keys())

        # Iterate over each experiment
        for exp_id in range(len(self.experiment_imgs)):
            num_frames = len(self.experiment_imgs[exp_id])

            for key in keys_to_filter:
                if key in self.result_data:
                    # Get specific filter settings for this key
                    window_length = filter_configs[key]['window_length']
                    polyorder = filter_configs[key]['polyorder']

                    # The window_length must be smaller than the number of data points
                    if window_length > num_frames:
                        logger.warning(f"Experiment {exp_id}, key '{key}': Cannot apply filter because window_length ({window_length}) > num_frames ({num_frames}). Skipping.")
                        continue

                    data_series = self.result_data[key][exp_id]
                    
                    # Sanitize data to handle inhomogeneous shapes
                    if any(isinstance(el, (list, tuple)) for el in data_series):
                        ref_len = None
                        for el in data_series:
                            if isinstance(el, (list, tuple)) and el:
                                ref_len = len(el)
                                break
                        
                        if ref_len is None:
                            logger.warning(f"Experiment {exp_id}, key '{key}': No valid data found to determine dimension for filtering. Skipping.")
                            continue
                            
                        sanitized_series = []
                        for el in data_series:
                            if isinstance(el, (list, tuple)) and len(el) == ref_len:
                                sanitized_series.append(el)
                            else:
                                sanitized_series.append([0.0] * ref_len)
                        
                        original_data = np.array(sanitized_series)
                    else:
                        original_data = np.array(data_series)

                    # Check if data is non-empty
                    if original_data.size > 0:
                        # Apply the filter along the time axis (axis=0)
                        filtered_data = savgol_filter(original_data, window_length, polyorder, axis=0)
                        
                        # Replace the old data with the new smoothed data
                        self.result_data[key][exp_id] = filtered_data.tolist()
        
        logger.success("Finished applying filter.")
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # --- END OF NEW METHOD ---


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
                    exp_name = self.experiment_names[exp_id]
                    for img_idx, img in enumerate(experiment_group_imgs):
                        if img is not None:
                            target_dir = os.path.join(base_output_dir, exp_name, result_type)
                            os.makedirs(target_dir, exist_ok=True)
                            
                            file_path = os.path.join(target_dir, f"{img_idx:03d}.png")
                            cv2.imwrite(file_path, img)
                            
                            pbar.update(1)
        
        logger.success("finished")

    def save_result_data(self, output_csv_path: str):
        logger.info(f"Aggregating and saving result data to '{output_csv_path}'...")
        
        all_data = []
        
        # Iterate through each experiment and each frame to build a flat list of data records
        for exp_id, exp_frames in enumerate(self.experiment_imgs):
            condition = self.experiment_conditions[exp_id]
            condition_data = asdict(condition)
            exp_name = self.experiment_names[exp_id]

            for frame_id in range(len(exp_frames)):
                # Start with the experiment condition and name
                data = condition_data.copy()
                data['experiment_name'] = exp_name
                
                # Add all the calculated data for the current frame
                for key in self.result_data:
                    # Exclude data points that are not easily serializable to CSV
                    if key not in ['contour_pts']:
                        data[key] = self.result_data[key][exp_id][frame_id]

                all_data.append(data)

        if not all_data:
            logger.warning("No data was aggregated. The output CSV will not be created.")
            return

        # Convert the list of dictionaries to a pandas DataFrame
        df = pd.DataFrame(all_data)
        
        # Ensure 'experiment_name' and 'frame_id' are the first columns for clarity
        cols = ['experiment_name', 'frame_id'] + [col for col in df.columns if col not in ['experiment_name', 'frame_id']]
        df = df[cols]

        # Save the DataFrame to a single CSV file
        try:
            output_dir = os.path.dirname(output_csv_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Check if file exists to decide whether to write header
            file_exists = os.path.exists(output_csv_path)
            df.to_csv(output_csv_path, mode='a', header=not file_exists, index=False)
            logger.success(f"Successfully saved {len(all_data)} records to {output_csv_path}")
        except IOError as e:
            logger.error(f"Could not write to CSV file {output_csv_path}: {e}")


class DataManager:
    def __init__(self, config: Config):
        self.config: dict = config.get_config()
        self.img_input_folder: str = self.config['img']['img_input_dir']
        self.img_output_folder: str = self.config['img']['img_output_dir']
        self.result_output_folder: str = self.config['img']['result_output_dir']
        # --- Get the new CSV path from config ---
        self.result_csv_path: str = self.config['img'].get('result_csv_path', 'data/dataset/dataset.csv')
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
    
    def load_all_experiment(self, end_idx=-1):
        folders = os.listdir(self.img_input_folder)
        logger.debug(f"{len(folders)} folders found.")
        for folder_idx, folder in enumerate(folders):
            if end_idx != -1 and folder_idx == end_idx: break

            logger.debug(f"Begin loading experiment {folder}")

            experiment_path = os.path.join(self.img_input_folder, folder)
            files = os.listdir(experiment_path)
            files = [f for f in files if f.endswith('.jpg')]
            
            # --- FIX: Sort files numerically to ensure correct frame order ---
            files.sort(key=lambda f: int(os.path.splitext(f)[0].replace('frame_', '')))
            
            logger.debug(f"{len(files)} imgs found.") 

            condition = self.parse_condition(folder)
            logger.debug(f"condition = {condition}")
            self.experiment.add_experiment(condition, folder)

            for file in tqdm(files, desc=f"loading imgs from {folder}"):
                img = cv2.imread(os.path.join(experiment_path, file))
                self.experiment.add_experiment_img(img)
            
        self.experiment.init_result_imgs()
    
    def save_all_experiment(self):
        # --- 2. APPLY FILTER: Define custom filter settings for each key ---
        filter_configs = {
            'area': {'window_length': 199, 'polyorder': 5},
            'arc_length': {'window_length': 199, 'polyorder': 5},
            'area_vec': {'window_length': 399, 'polyorder': 5},
            'regression_circle_radius': {'window_length': 199, 'polyorder': 5},
            'expand_dist': {'window_length': 199, 'polyorder': 5},
            'expand_vec': {'window_length': 399, 'polyorder': 5},
            'tip_distance': {'window_length': 1999, 'polyorder': 5}, # Stronger filter for noisy data
        }
        self.experiment.apply_savitzky_golay_filter(filter_configs)
        
        # Now, save the newly smoothed data
        if self.config.get('img', {}).get('save_processed_images', True):
            self.experiment.save_result_imgs(self.img_output_folder)
        
        if self.config.get('img', {}).get('save_processed_data', True):
            self.experiment.save_result_data(self.result_csv_path)