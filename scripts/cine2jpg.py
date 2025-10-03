import os
import glob
import cv2  
from pycine.raw import read_frames

# --- Configuration Section ---
CINE_SOURCE_DIR = os.path.join('data', 'raw_cine')
JPG_TARGET_DIR = os.path.join('data', 'raw_jpg')
# --- End of Configuration ---

def process_single_cine(cine_filepath, output_base_dir):
    cine_filename = os.path.basename(cine_filepath)
    frame_cnt = 0
    print(f"--- Processing: {cine_filename} ---")

    try:
        # 1. 
        folder_name = os.path.splitext(cine_filename)[0]
        final_output_path = os.path.join(output_base_dir, folder_name)
        os.makedirs(final_output_path, exist_ok=True)
        print(f"Images will be saved to: {final_output_path}")

        # 2. 
        raw_images, _, _ = read_frames(cine_filepath, start_frame=0, count=None)

        for i, img in enumerate(raw_images):
            frame_cnt += 1
            output_filename = f"frame_{i:05d}.jpg"
            full_save_path = os.path.join(final_output_path, output_filename)
            _,ret = cv2.imencode('.jpg', img) 
            with open(full_save_path, 'wb') as f:
                f.write(ret)

        print(f"Successfully converted and saved {frame_cnt} frames.")

    except Exception as e:
        print(f"An error occurred while processing {cine_filename}: {e}")

def main():
    print("Starting batch conversion from CINE to JPG...")
    search_pattern = os.path.join(CINE_SOURCE_DIR, '*.cine')
    cine_files = glob.glob(search_pattern)

    if not cine_files:
        print(f"No .cine files found in '{CINE_SOURCE_DIR}'. Please check the path.")
        return

    print(f"Found {len(cine_files)} .cine file(s) to process.")

    for cine_file in cine_files:
        process_single_cine(cine_file, JPG_TARGET_DIR)
        print("-" * 40)

    print("All conversion tasks are complete!")

if __name__ == '__main__':
    os.makedirs(JPG_TARGET_DIR, exist_ok=True)
    main()