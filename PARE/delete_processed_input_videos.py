import os

# Define paths
output_dir = '/x4/ljiao/Bhanu/PARE/output'
input_dir = '/x4/ljiao/Bhanu/PARE/input'

# Mandatory files to verify processing
required_files = ['demo.log', 'pare_output.pkl', 'tmp_images', 'tmp_images_output']

# Iterate through input directories
for subfolder in os.listdir(input_dir):
    subfolder_path = os.path.join(input_dir, subfolder)
    
    if os.path.isdir(subfolder_path):
        for video_file in os.listdir(subfolder_path):
            if video_file.endswith('.mp4'):
                video_name = os.path.splitext(video_file)[0]

                # Construct the corresponding output folder path
                output_folder = os.path.join(output_dir, video_name, f'{video_name}_')

                # Check if the output folder and required files exist
                if os.path.exists(output_folder):
                    output_files = os.listdir(output_folder)

                    # Check if all required files are present
                    if all(req_file in output_files for req_file in required_files) and \
                       any(f.startswith(video_name) and f.endswith('.mp4') for f in output_files) and \
                       any(f.startswith(video_name) and f.endswith('__result.mp4') for f in output_files):
                        
                        # Delete the input video
                        video_path = os.path.join(subfolder_path, video_file)
                        try:
                            os.remove(video_path)
                            print(f"Deleted: {video_path}")
                        except Exception as e:
                            print(f"Error deleting {video_path}: {e}")
