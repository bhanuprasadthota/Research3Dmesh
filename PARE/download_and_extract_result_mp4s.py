import os
import shutil
import zipfile
import requests
from huggingface_hub import HfApi, hf_hub_url, list_repo_files

# ---- CONFIG ----
REPO_ID = "ThotaBhanu/PAREProcessed"
REPO_TYPE = "dataset"
ZIP_FOLDER = "videos"
SAVE_DIR = "/x4/ljiao/Bhanu/PARE/all_result_videos"         # Where __result.mp4 files go
TEMP_DIR = "/x4/ljiao/Bhanu/PARE/temp_zip_extract"          # Temporary unzip folder
TOKEN = os.getenv("HUGGINGFACE_TOKEN")                      # Optional for private repos

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# ---- DOWNLOAD + PROCESS ----
api = HfApi()
files = list_repo_files(REPO_ID, repo_type=REPO_TYPE)
zip_files = [f for f in files if f.startswith(ZIP_FOLDER) and f.endswith(".zip")]
#zip_files = [f for f in files if f.endswith(".zip")]


for zip_path in zip_files:
    filename = os.path.basename(zip_path)
    zip_url = hf_hub_url(REPO_ID, filename=zip_path, repo_type=REPO_TYPE)

    print(f"🔽 Downloading: {filename}...")
    response = requests.get(zip_url, stream=True)
    local_zip_path = os.path.join(TEMP_DIR, filename)

    with open(local_zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"📦 Extracting: {filename}...")
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(TEMP_DIR)

    print("📂 Moving __result.mp4 files...")
    for root, _, files in os.walk(TEMP_DIR):
        for f in files:
            if f.endswith("__result.mp4"):
                full_path = os.path.join(root, f)
                target_path = os.path.join(SAVE_DIR, f)
                if not os.path.exists(target_path):
                    shutil.move(full_path, target_path)

    print("🧹 Cleaning up temp files...")

    # ✅ Delete the ZIP file *before* deleting the folder
    os.remove(local_zip_path)

    # ✅ Clear temp unzip contents
    shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)

print("✅ All zip files processed and __result.mp4 files extracted.")
