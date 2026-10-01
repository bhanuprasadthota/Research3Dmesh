import os
import zipfile
from huggingface_hub import HfApi

output_dir = '/x4/ljiao/Bhanu/PARE/output'
zip_dir = '/x4/ljiao/Bhanu/PARE/zipped'
os.makedirs(zip_dir, exist_ok=True)

api = HfApi()
repo_id = "ThotaBhanu/PAREProcessed"

max_size_gb = 45
part_counter = 5

while True:
    zip_file_path = os.path.join(zip_dir, f'processed_part{part_counter}.zip')
    log_file_path = os.path.join(zip_dir, f'zipped_folders_part{part_counter}.txt')
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

    if not folders_to_zip:
        print("No more folders to process. Exiting...")
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

    # Upload to Hugging Face
    print(f"Uploading {zip_file_path} to Hugging Face...")
    api.upload_file(
        path_or_fileobj=zip_file_path,
        path_in_repo=os.path.basename(zip_file_path),
        repo_id=repo_id,
        repo_type="dataset"
    )
    print(f"Upload completed: {zip_file_path}")

    # Delete the processed folders
    with open(log_file_path, 'r') as log_file:
        for folder_name in log_file:
            folder_name = folder_name.strip()
            folder_path = os.path.join(output_dir, folder_name)
            if os.path.exists(folder_path):
                print(f"Deleting folder: {folder_path}")
                for root, _, files in os.walk(folder_path, topdown=False):
                    for file in files:
                        os.remove(os.path.join(root, file))
                    os.rmdir(root)

    # Delete the zip file
    os.remove(zip_file_path)
    print(f"Deleted zip file: {zip_file_path}")

    part_counter += 1

print("Process completed.")
