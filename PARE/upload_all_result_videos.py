from huggingface_hub import HfApi

api = HfApi()

repo_id = "ThotaBhanu/PAREProcessed"
file_path = "/x4/ljiao/Bhanu/PARE/all_result_videos.zip"

api.upload_file(
    path_or_fileobj=file_path,
    path_in_repo="videos/all_result_videos.zip",
    repo_id=repo_id,
    repo_type="dataset"
)

print("✅ Upload complete!")
