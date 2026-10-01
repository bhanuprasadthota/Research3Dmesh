import os
import subprocess
import time

input_dir = "/x4/ljiao/Bhanu/PARE/input/kungfu_videos"
output_dir = "/x4/ljiao/Bhanu/PARE/output"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

video_files = sorted(
    [f for f in os.listdir(input_dir) if f.lower().endswith('.mp4')],
    key=lambda x: os.path.getsize(os.path.join(input_dir, x))
)

total_videos = len(video_files)
success_count = 0
skipped_count = 0
error_count = 0
timings = []

print(f"🎯 Found {total_videos} video(s) to process.\n")

for idx, video in enumerate(video_files, start=1):
    input_path = os.path.join(input_dir, video)
    video_name = os.path.splitext(video)[0]
    output_path = os.path.join(output_dir, video_name)

    # Check if the folder already exists
    if os.path.exists(output_path):
        print(f"[{idx}/{total_videos}] ⏭ Skipping '{video}' (already processed)")
        skipped_count += 1
        continue

    os.makedirs(output_path, exist_ok=True)

    print(f"[{idx}/{total_videos}] 🔄 Processing '{video}'...")

    start_time = time.time()
    try:
        subprocess.run([
            "python", "scripts/demo.py",
            "--vid_file", input_path,
            "--output_folder", output_path
        ], check=True)
        elapsed = time.time() - start_time
        timings.append(elapsed)
        success_count += 1
        print(f"[{idx}/{total_videos}] ✅ Done in {elapsed:.2f}s: '{video}'")

        # Estimate remaining time
        remaining = total_videos - idx
        avg_time = sum(timings) / len(timings)
        est_total_remaining = remaining * avg_time
        print(f"⏳ Estimated time left: {est_total_remaining/60:.2f} minutes\n")

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"[{idx}/{total_videos}] ❌ Error after {elapsed:.2f}s: '{video}'\n")
        error_count += 1

# Final Summary
print("🧾 Summary:")
print(f"✅ Processed: {success_count}")
print(f"⏭ Skipped  : {skipped_count}")
print(f"❌ Errors   : {error_count}")
print(f"🕒 Avg Time : {sum(timings)/len(timings):.2f}s per video" if timings else "")
