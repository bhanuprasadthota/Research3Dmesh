import os
import subprocess

# Path to your data folder containing S11_01, S11_02, ...
DATA_DIR = './vibe_data/PIN3DWIFI'
# Output base directory
OUTPUT_BASE = './output' # Change this directory

# Loop over all subdirectories in the data folder
for folder_name in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, folder_name)
    video_path = os.path.join(folder_path, 'output.mkv')

    if os.path.isdir(folder_path) and os.path.isfile(video_path):
        output_folder = os.path.join(OUTPUT_BASE, folder_name)
        os.makedirs(output_folder, exist_ok=True)

        print(f"Processing {video_path} -> {output_folder}")

        cmd = f'CUDA_VISIBLE_DEVICES=1 python demo.py --vid_file "{video_path}" --output_folder "{output_folder}"'
        subprocess.run(cmd, shell=True)


print("All videos processed.")
