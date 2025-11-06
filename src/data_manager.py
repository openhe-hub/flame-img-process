import cv2
import os
import re
import shutil
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
        non_negative_keys = {'area', 'arc_length', 'regression_circle_radius', 'tip_distance', 'expand_dist'}

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
                        series_length = original_data.shape[0]
                        if series_length <= polyorder:
                            logger.warning(
                                f"Experiment {exp_id}, key '{key}': series length ({series_length}) "
                                f"is not greater than polyorder ({polyorder}). Skipping."
                            )
                            continue

                        effective_window = min(window_length, series_length)
                        if effective_window % 2 == 0:
                            effective_window -= 1

                        if effective_window <= polyorder:
                            logger.warning(
                                f"Experiment {exp_id}, key '{key}': adjusted window_length ({effective_window}) "
                                f"is not greater than polyorder ({polyorder}). Skipping."
                            )
                            continue

                        if effective_window != window_length:
                            logger.debug(
                                f"Experiment {exp_id}, key '{key}': window_length reduced from "
                                f"{window_length} to {effective_window} to fit series length ({series_length})."
                            )

                        # Apply the filter along the time axis (axis=0)
                        filtered_data = savgol_filter(original_data, effective_window, polyorder, axis=0)

                        if key in non_negative_keys:
                            filtered_data = np.maximum(filtered_data, 0.0)
                        
                        # Replace the old data with the new smoothed data
                        self.result_data[key][exp_id] = filtered_data.tolist()
        
        logger.success("Finished applying filter.")
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # --- END OF NEW METHOD ---


    def save_result_imgs(self, base_output_dir: str):
        logger.info(f"saving result imgs to '{base_output_dir}'...")

        # Remove stale outputs for current experiments to avoid leftover frames beyond the trimmed range.
        if os.path.isdir(base_output_dir):
            for exp_name in self.experiment_names:
                exp_dir = os.path.join(base_output_dir, exp_name)
                if os.path.isdir(exp_dir):
                    shutil.rmtree(exp_dir)

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
        logger.info("Saving result data per experiment...")

        img_config = self.config.get('img', {})
        output_dir = img_config.get('result_output_dir')
        if not output_dir:
            output_dir = os.path.dirname(output_csv_path)
        if not output_dir:
            output_dir = "."

        os.makedirs(output_dir, exist_ok=True)

        # Remove legacy aggregated file if it exists to avoid confusion.
        if output_csv_path and os.path.isfile(output_csv_path):
            try:
                os.remove(output_csv_path)
                logger.info(f"Removed legacy aggregated CSV at '{output_csv_path}'.")
            except OSError as exc:
                logger.warning(f"Unable to remove legacy CSV '{output_csv_path}': {exc}")

        for exp_id, _ in enumerate(self.experiment_imgs):
            condition = self.experiment_conditions[exp_id]
            condition_data = asdict(condition)
            exp_name = self.experiment_names[exp_id]

            frame_ids = self.result_data['frame_id'][exp_id]
            frame_count = len(frame_ids)
            if frame_count == 0:
                logger.warning(f"Experiment '{exp_name}' has no frames after trimming; skipping CSV export.")
                continue

            exp_records = []
            for frame_idx in range(frame_count):
                record = condition_data.copy()
                record['experiment_name'] = exp_name

                for key, per_experiment_data in self.result_data.items():
                    if key == 'contour_pts':
                        continue
                    record[key] = per_experiment_data[exp_id][frame_idx]

                exp_records.append(record)

            if not exp_records:
                logger.warning(f"No data collected for experiment '{exp_name}'. Skipping export.")
                continue

            df = pd.DataFrame(exp_records)
            columns_order = ['experiment_name', 'frame_id'] + [
                col for col in df.columns if col not in ['experiment_name', 'frame_id']
            ]
            df = df[columns_order]

            exp_csv_path = os.path.join(output_dir, f"{exp_name}.csv")
            try:
                df.to_csv(exp_csv_path, index=False)
                logger.success(f"Saved {len(df)} records to '{exp_csv_path}'.")
            except IOError as exc:
                logger.error(f"Could not write CSV for experiment '{exp_name}' to '{exp_csv_path}': {exc}")


class DataManager:
    def __init__(self, config: Config):
        self.config: dict = config.get_config()
        self.img_input_folder: str = self.config['img']['img_input_dir']
        self.img_output_folder: str = self.config['img']['img_output_dir']
        self.result_output_folder: str = self.config['img']['result_output_dir']
        # --- Get the new CSV path from config ---
        self.result_csv_path: str = self.config['img'].get('result_csv_path', 'data/dataset/dataset.csv')
        self.experiment = Experiment(self.config)
        self.experiment_groups: Dict[str, List[int]] = {}
        self.group_order: List[str] = []

    @staticmethod
    def normalize_experiment_name(name: str) -> str:
        return re.sub(r'-\d+$', '', name)

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

            base_name = self.normalize_experiment_name(folder)
            if base_name not in self.experiment_groups:
                self.experiment_groups[base_name] = []
                self.group_order.append(base_name)
            self.experiment_groups[base_name].append(self.experiment.curr_experiment_idx)

            for file in tqdm(files, desc=f"loading imgs from {folder}"):
                img = cv2.imread(os.path.join(experiment_path, file))
                self.experiment.add_experiment_img(img)
            
        self.experiment.init_result_imgs()

    def aggregate_replicates(self):
        if not self.experiment_groups:
            return

        exp = self.experiment
        new_experiment_imgs: List[List[cv2.Mat]] = []
        new_experiment_features: List[List[Feature]] = []
        new_experiment_conditions: List[ExperimentCondition] = []
        new_experiment_names: List[str] = []

        new_result_imgs = {key: [] for key in exp.result_imgs}
        new_result_data = {key: [] for key in exp.result_data}

        updated_groups: Dict[str, List[int]] = {}
        aggregated_any = False

        def _prepare_numeric_sequence(seq, length):
            trimmed = seq[:length]
            first_shape = None
            for elem in trimmed:
                if isinstance(elem, (list, tuple, np.ndarray)):
                    arr = np.asarray(elem, dtype=float)
                    if arr.size == 0:
                        continue
                    first_shape = arr.shape
                    break
                elif elem is not None:
                    first_shape = ()
                    break
            if first_shape is None:
                first_shape = ()

            if first_shape == ():
                try:
                    return np.asarray(trimmed, dtype=float)
                except ValueError:
                    sanitized = []
                    for elem in trimmed:
                        try:
                            sanitized.append(float(elem))
                        except (TypeError, ValueError):
                            sanitized.append(0.0)
                    return np.asarray(sanitized, dtype=float)

            target_shape = first_shape
            sanitized = []
            zeros_template = np.zeros(target_shape, dtype=float)
            for elem in trimmed:
                if isinstance(elem, (list, tuple, np.ndarray)):
                    arr = np.asarray(elem, dtype=float)
                    if arr.shape == target_shape:
                        sanitized.append(arr)
                        continue
                sanitized.append(zeros_template.copy())
            return np.stack(sanitized, axis=0)

        for base_name in self.group_order:
            indices = self.experiment_groups.get(base_name, [])
            if not indices:
                continue

            indices = sorted(indices)
            representative_idx = indices[0]

            sequence_lengths = [len(exp.result_data['time'][idx]) for idx in indices if len(exp.result_data['time'][idx]) > 0]
            if not sequence_lengths:
                logger.warning(f"Experiment group '{base_name}' has no valid frames; skipping.")
                continue

            min_len = min(sequence_lengths)
            if len(indices) > 1:
                aggregated_any = True
                logger.info(f"Aggregating {len(indices)} replicates for experiment '{base_name}' (using first {min_len} frames).")

            # Raw images and features (use representative replicate trimmed to min_len)
            new_experiment_imgs.append(exp.experiment_imgs[representative_idx][:min_len])
            if exp.experiment_features[representative_idx]:
                new_experiment_features.append(exp.experiment_features[representative_idx][:min_len])
            else:
                new_experiment_features.append([])
            new_experiment_conditions.append(exp.experiment_conditions[representative_idx])
            new_index = len(new_experiment_names)
            new_experiment_names.append(base_name)
            updated_groups[base_name] = [new_index]

            # Result images: prefer first non-None frame among replicates
            for key in new_result_imgs:
                frames: List[cv2.Mat] = []
                for frame_idx in range(min_len):
                    chosen_frame = None
                    for idx in indices:
                        frames_list = exp.result_imgs[key][idx]
                        if frame_idx < len(frames_list):
                            candidate = frames_list[frame_idx]
                            if candidate is not None:
                                chosen_frame = candidate
                                break
                    frames.append(chosen_frame)
                new_result_imgs[key].append(frames)

            # Result data aggregation (averages for numeric data, representative for identifiers)
            for key in new_result_data:
                if key == 'contour_pts':
                    new_result_data[key].append(exp.result_data[key][representative_idx][:min_len])
                    continue
                if key in ('frame_id', 'time'):
                    new_result_data[key].append(exp.result_data[key][representative_idx][:min_len])
                    continue

                sequences = []
                for idx in indices:
                    seq = exp.result_data[key][idx]
                    prepared = _prepare_numeric_sequence(seq, min_len)
                    sequences.append(prepared)

                if not sequences:
                    new_result_data[key].append([])
                    continue

                stacked = np.stack(sequences, axis=0)
                averaged = stacked.mean(axis=0)
                new_result_data[key].append(averaged.tolist())

        if aggregated_any:
            logger.success("Finished aggregating replicate experiments.")

        if not new_experiment_names:
            logger.warning("No experiments available after replicate aggregation; retaining original data.")
            return

        exp.experiment_imgs = new_experiment_imgs
        exp.experiment_features = new_experiment_features
        exp.experiment_conditions = new_experiment_conditions
        exp.experiment_names = new_experiment_names
        exp.experiment_cnt = len(new_experiment_names)
        exp.curr_experiment_idx = len(new_experiment_names) - 1
        exp.result_imgs = new_result_imgs
        exp.result_data = new_result_data

        self.experiment_groups = updated_groups
        self.group_order = new_experiment_names

    def save_all_experiment(self):
        # --- 2. APPLY FILTER: Define custom filter settings for each key ---
        filter_configs = {
            'area': {'window_length': 199, 'polyorder': 5},
            'arc_length': {'window_length': 199, 'polyorder': 5},
            'area_vec': {'window_length': 399, 'polyorder': 5},
            'regression_circle_radius': {'window_length': 199, 'polyorder': 5},
            'expand_dist': {'window_length': 199, 'polyorder': 5},
            'expand_vec': {'window_length': 399, 'polyorder': 5},
            'tip_distance': {'window_length': 999, 'polyorder': 5}, # Stronger filter for noisy data
        }
        self.experiment.apply_savitzky_golay_filter(filter_configs)
        self.aggregate_replicates()
        
        # Now, save the newly smoothed data
        if self.config.get('img', {}).get('save_processed_images', True):
            self.experiment.save_result_imgs(self.img_output_folder)
        
        if self.config.get('img', {}).get('save_processed_data', True):
            self.experiment.save_result_data(self.result_csv_path)
