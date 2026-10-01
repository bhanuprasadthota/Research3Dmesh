#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 02:54:47 2025

@author: bhanuprasadthota
"""

import cv2
import os
import shutil
from ultralytics import YOLO
import mediapipe as mp

# Paths
SOURCE_FOLDER = "/x1/ljiao/Bhanu/VIBE/Project/input/idea400"
DESTINATION_FOLDER_VALID = "/x1/ljiao/Bhanu/VIBE/cAvatar/idea400"
DESTINATION_FOLDER_INVALID = "/x1/ljiao/Bhanu/VIBE/Project/input/idea400/rejected_videos"

# Create folders if they don't exist
os.makedirs(DESTINATION_FOLDER_VALID, exist_ok=True)
os.makedirs(DESTINATION_FOLDER_INVALID, exist_ok=True)

# Load YOLOv8 Model (for person detection)
model = YOLO("yolov8n.pt")

# Load MediaPipe Pose (for full-body check)
mp_pose = mp.solutions.pose
pose_estimator = mp_pose.Pose(static_image_mode=True, enable_segmentation=False,
                              min_detection_confidence=0.4, model_complexity=1)

# Function to check if person is fully visible
def is_full_body_visible(frame, bbox, pose_results):
    frame_h, frame_w = frame.shape[:2]

    # Bounding box check - ensure body isn’t touching edges (cropped)
    x1, y1, x2, y2 = bbox
    margin = 5  # pixels tolerance for edges
    if x1 < margin or y1 < margin or x2 > frame_w - margin or y2 > frame_h - margin:
        print("❌ Bounding box touches frame edge — possible cropped body.")
        return False  

    # Pose check for full-body keypoints (head, ankles)
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

        # Check visibility score and position for key body parts
        if left_ankle.visibility < 0.4 or right_ankle.visibility < 0.4:
            print("❌ Ankles not clearly visible — possible cropped lower body.")
            return False  
        if nose.visibility < 0.4 or nose.y < 0.05:
            print("❌ Nose not clearly visible — possible cropped head.")
            return False  

    return True  # All checks passed - full body confirmed

# Main processing loop
valid_videos = []
invalid_videos = []

for video_file in os.listdir(SOURCE_FOLDER):
    if not video_file.endswith(".mp4"):
        continue  # Skip non-mp4 files

    video_path = os.path.join(SOURCE_FOLDER, video_file)
    cap = cv2.VideoCapture(video_path)

    # Sampling strategy: 1 frame every 2 seconds
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = int(fps * 2)  # Sample every 2 seconds
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_passes = True  # Assume the video is valid until proven otherwise
    reason_for_rejection = ""

    # Process sampled frames
    frame_idx = 0
    while frame_idx < frame_count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Detect persons using YOLOv8
        results = model(frame, verbose=False)
        det = results[0]
        persons = [
            box.xyxy[0] for box in det.boxes
            if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.5
        ]

        # Criteria 1: Must have exactly one person in frame
        if len(persons) != 1:
            video_passes = False
            reason_for_rejection = f"❌ Found {len(persons)} person(s) in frame {frame_idx}."
            break

        # Criteria 2: Full-body check (pose landmarks)
        bbox = persons[0]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose_estimator.process(rgb_frame)

        if not is_full_body_visible(frame, bbox, pose_results):
            video_passes = False
            reason_for_rejection = f"❌ Partial body detected in frame {frame_idx}."
            break

        frame_idx += interval  # Move to the next sampled frame

    cap.release()

    # Save videos based on result
    if video_passes:
        valid_videos.append(video_file)
        shutil.copy(video_path, os.path.join(DESTINATION_FOLDER_VALID, video_file))
    else:
        invalid_videos.append(video_file)
        shutil.copy(video_path, os.path.join(DESTINATION_FOLDER_INVALID, video_file))
        print(f"{reason_for_rejection} Video '{video_file}' rejected.")

print(f"✅ Filtered {len(valid_videos)} valid videos saved in '{DESTINATION_FOLDER_VALID}'")
print(f"❌ Rejected {len(invalid_videos)} invalid videos saved in '{DESTINATION_FOLDER_INVALID}'")
print("Valid Videos:", valid_videos)
print("Invalid Videos:", invalid_videos)