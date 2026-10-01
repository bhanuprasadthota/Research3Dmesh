import os
import subprocess

# Define input and output folders
input_folder = "/x1/ljiao/Bhanu/VIBE/cAvatar/idea400"
output_folder = "/x1/ljiao/Bhanu/VIBE/cAvatar/idea400"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# List all .mp4 files in the input folder
video_files = [file for file in os.listdir(input_folder) if file.endswith(".mp4")]

# Total number of videos
total_videos = len(video_files)
completed_videos = 0  # Counter for completed videos

# Iterate through all .mp4 files
for video_file in video_files:
    input_video_path = os.path.join(input_folder, video_file)
    
    # Run Mesh.py for each video
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




