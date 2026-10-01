import os
import subprocess

# Define input and output folders
input_folder = "/x1/ljiao/Bhanu/VIBE/cAvatar/dance"
output_folder = "/x1/ljiao/Bhanu/VIBE/cAvatar/dance"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# List all .mp4 files that don't end with '_vibe_result.mp4'
video_files = [
    file for file in os.listdir(input_folder)
    if file.endswith(".mp4") and not file.endswith("_vibe_result.mp4")
]

# Filter out videos that already have a corresponding .pkl file
videos_to_process = [
    file for file in video_files
    if not os.path.exists(os.path.join(output_folder, file.replace(".mp4", ".pkl")))
]

# Total number of videos to process
total_videos = len(videos_to_process)
completed_videos = 0

# Iterate through filtered list
for video_file in videos_to_process:
    input_video_path = os.path.join(input_folder, video_file)

    try:
        subprocess.run([
            "python", "onlymeshvideo.py",
            "--vid_file", input_video_path,
            "--output_folder", output_folder
        ], check=True)

        completed_videos += 1
        print(f"✅ Processed: {video_file} ({completed_videos}/{total_videos} videos completed)")

    except subprocess.CalledProcessError:
        print(f"❌ Failed to process: {video_file}")

print("🎯 Batch processing complete!")
