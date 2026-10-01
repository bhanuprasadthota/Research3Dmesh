import os
import zipfile

output_dir = '/x4/ljiao/Bhanu/PARE/output'
zip_dir = '/x4/ljiao/Bhanu/PARE/zipped'
os.makedirs(zip_dir, exist_ok=True)

zip_file_path = os.path.join(zip_dir, 'processed_part4.zip')
log_file_path = os.path.join(zip_dir, 'zipped_folders_part4.txt')

max_size_gb = 45
current_size_gb = 0
folders_to_zip = []

# Collect folders and calculate size
for folder in sorted(os.listdir(output_dir)):
    folder_path = os.path.join(output_dir, folder)
    if os.path.isdir(folder_path):
        folder_size = sum(os.path.getsize(os.path.join(root, file)) for root, _, files in os.walk(folder_path) for file in files)
        folder_size_gb = folder_size / (1024 ** 3)
        if current_size_gb + folder_size_gb <= max_size_gb:
            folders_to_zip.append(folder)
            current_size_gb += folder_size_gb
        if current_size_gb >= max_size_gb:
            break

# Create the zip file and log the zipped folders
with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    with open(log_file_path, 'w') as log_file:
        for folder in folders_to_zip:
            folder_path = os.path.join(output_dir, folder)
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)
            log_file.write(f"{folder}\n")

print(f"Zipping completed. Zip file saved at: {zip_file_path}")
print(f"Log file created at: {log_file_path}")
