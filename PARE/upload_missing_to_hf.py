from huggingface_hub import HfApi

api = HfApi()

repo_id = "ThotaBhanu/PAREProcessed"
file_path = "/x4/ljiao/Bhanu/PARE/videos_result_missing.zip"

api.upload_file(
    path_or_fileobj=file_path,
    path_in_repo="videos/videos_result_missing.zip",
    repo_id=repo_id,
    repo_type="dataset"
)

print("✅ Upload complete!")
