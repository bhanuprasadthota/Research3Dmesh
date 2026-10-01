import os
import shutil

def process_videos(input_dir, output_dir):
    # Counter to keep track of how many files were processed
    processed_count = 0

    # Walk through all subdirectories in the input directory
    for root, dirs, files in os.walk(input_dir, topdown=False):  # topdown=False to avoid errors while deleting subfolders
        # Skip directories that don't contain a `.pkl` file
        if not any(f.endswith('.pkl') for f in files):
            continue

        # Check if there is an .mp4 file in the folder's name (or inside the folder)
        if not any(f.endswith('.mp4') for f in files):
            continue

        # Look for the .pkl file and the .mp4 file in the folder
        pkl_file = [f for f in files if f.endswith('.pkl')][0]
        mp4_file = [f for f in files if f.endswith('.mp4')][0]

        # Get the new name for the .pkl file to match the .mp4 file
        new_pkl_filename = mp4_file.replace('.mp4', '.pkl')
        pkl_file_path = os.path.join(root, pkl_file)
        new_pkl_path = os.path.join(root, new_pkl_filename)

        # Rename the .pkl file to match the .mp4 file
        os.rename(pkl_file_path, new_pkl_path)

        # Move the .mp4 and renamed .pkl files to the output directory
        mp4_file_path = os.path.join(root, mp4_file)
        shutil.move(mp4_file_path, os.path.join(output_dir, mp4_file))
        shutil.move(new_pkl_path, os.path.join(output_dir, new_pkl_filename))

        # Delete the empty folder after moving the files
        if not os.listdir(root):  # Check if the directory is empty
            os.rmdir(root)

        # Increment the processed count
        processed_count += 1

    print(f"Processed {processed_count} videos.")

# Define your input and output directories
input_dir = '/x1/ljiao/Bhanu/VIBE/Project/output/dance/video_splits1'
output_dir = '/x1/ljiao/Bhanu/VIBE/Project/output/dance/video_splits1'

# Call the function
process_videos(input_dir, output_dir)
