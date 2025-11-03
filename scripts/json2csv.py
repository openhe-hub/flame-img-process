import os
import json
import pandas as pd
from tqdm import tqdm
from loguru import logger

def aggregate_json_to_csv(root_dir: str, output_csv_path: str):
    """
    Aggregates all .json files from subdirectories of a root directory into a single CSV file.

    Args:
        root_dir (str): The path to the root directory containing experiment subfolders.
        output_csv_path (str): The path where the output CSV file will be saved.
    """
    if not os.path.isdir(root_dir):
        logger.error(f"Root directory not found: {root_dir}")
        return

    all_data = []
    
    # Get a list of all experiment directories
    try:
        experiment_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
        if not experiment_folders:
            logger.warning(f"No subdirectories found in {root_dir}")
            return
    except FileNotFoundError:
        logger.error(f"Could not access directory: {root_dir}")
        return

    logger.info(f"Found {len(experiment_folders)} experiment folders. Starting aggregation...")

    # Iterate over each experiment folder with a progress bar
    for exp_name in tqdm(experiment_folders, desc="Aggregating Experiments", unit="folder"):
        exp_path = os.path.join(root_dir, exp_name)
        
        # Find all json files in the directory
        json_files = [f for f in os.listdir(exp_path) if f.endswith('.json')]
        
        # Sort files to ensure chronological order of frames
        json_files.sort()

        # Inner progress bar for files within the current experiment
        for json_file in tqdm(json_files, desc=f"  -> Processing {exp_name[:30]:<30}", unit="file", leave=False):
            frame_id = os.path.splitext(json_file)[0]
            file_path = os.path.join(exp_path, json_file)
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Remove the 'contour_pts' field if it exists
                data.pop('contour_pts', None)

                # Add the experiment name and frame_id to the data
                data['experiment_name'] = exp_name
                # The frame_id from the JSON is more reliable, but we can use the filename as a fallback
                if 'frame_id' not in data:
                    data['frame_id'] = int(frame_id)

                all_data.append(data)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not read or parse {file_path}: {e}")

    if not all_data:
        logger.warning("No data was aggregated. The output CSV will not be created.")
        return

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(all_data)
    
    # Ensure 'experiment_name' and 'frame_id' are the first columns for clarity
    cols = ['experiment_name', 'frame_id'] + [col for col in df.columns if col not in ['experiment_name', 'frame_id']]
    df = df[cols]

    # Save the DataFrame to a CSV file
    try:
        output_dir = os.path.dirname(output_csv_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        logger.success(f"Successfully aggregated {len(all_data)} records into {output_csv_path}")
    except IOError as e:
        logger.error(f"Could not write to CSV file {output_csv_path}: {e}")


if __name__ == '__main__':
    # --- Configuration ---
    # The root directory where your processed experiment folders are located.
    DATA_ROOT = 'data/dataset'
    # The path for the final aggregated CSV file.
    OUTPUT_CSV = 'data/dataset/dataset.csv'
    
    aggregate_json_to_csv(DATA_ROOT, OUTPUT_CSV)
